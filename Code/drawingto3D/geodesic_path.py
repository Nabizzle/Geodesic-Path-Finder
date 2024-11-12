from tkinter import messagebox
import numpy as np
import polars as pl
import progressbar
from progressbar import ETA, GranularBar, Percentage
import cv2
from scipy.spatial import KDTree
from typing import Dict
from tkinter.filedialog import askopenfilename
from drawingto3D.data_manager import load_mesh
from potpourri3d import compute_distance as compute_distance
from pathos.multiprocessing import ProcessingPool as Pool
import time


def solve(mesh_vertices, mesh_faces, start_ind, end_ind):
    dist = compute_distance(mesh_vertices, mesh_faces, start_ind)
    distance = dist[end_ind]
    return distance


class GeodesicPath():
    '''
    Finds the geodesic path between sets of points on a mesh

    After selecting the side and sex of the model, the user can input data
    through function calls and get out calculated distance and path information
    from the geodesic path found on the mesh.

    Attributes
    ----------
    drawing_name : str
        The name of the 2D location drawing template
    mesh_name : str
        The name of the mesh to find the geodesic distance and path on
    distance_solver : MeshHeatMethodDistanceSolver
        MeshHeatMethodDistanceSolver object for finding geodesic distances
        using the heat method
    path_solver : EdgeFlipGeodesicSolver
        EdgeFlipGeodesicSolver object for showing the geodesic path between two
        points using edge flips
    uv_array : ndarray
        numpy array of all of the uv data values of a mesh
    lookup_data: DataFrame
        Polars DataFrame of the data that make up the faces of the mesh. This
        is made by referencing vertex, uv, and normal vector index values from
        the rest of the mesh data.
    start_x_location : float or ndarray
        The x pixel value of the starting location drawing centroid
    start_y_location : float or ndarray
        The y pixel value of the starting location drawing centroid
    end_x_location : float or ndarray
        The x pixel value of the starting location drawing centroid
    end_y_location : float or ndarray
        The y pixel value of the starting location drawing centroid
    path_verticies : ndarray
        An array of the start and end vertex numbers for the location drawing
        centriods
    found_distances : ndarray
        An array of all of the found geodesic distances
    found_paths : Dict[str, ndarray]
        A dictionary of each path found labeled by a path number

    Methods
    -------
    __init__(sex, side)
        Sets up the names of the data to load in
    analyze_data(data)
        Loads in data and analyzes it
    analyzed_data_from_csv
        Loads in points to measure between from a file
    calculate_distances()
        Find the distance between the starting and ending points
    calculate_paths()
        Finds the path between the start and end vertex
    load_data(data)
        Takes an Nx4 numpy array and converts it to start and end points
    uv_to_vertex(centroid_x, centroid_y, image_x_size, image_y_size)
        Converts location drawing pixel value to 3D vertex location
    '''
    def __init__(self, p: Pool, sex: str = "male",
                 side: str = "right") -> None:
        '''
        Sets up the names of the data to load in

        Parameters
        ----------
        sex : str, default: male
            The visual sex of the mesh (Male or Female)
        side : str, default: right
            The arm of the model (right or left)

        Raises
        ------
        KeyError
            If in unknown input is made
        '''
        # make all input data lowercase
        sex = sex.lower()
        side = side.lower()
        self.p = p

        if side == "right":
            self.drawing_name = "right arm.png"
            self.mesh_name = "Right Arm"
        elif side == "left":
            self.drawing_name = "left arm.png"
            self.mesh_name = "Left Arm"
        else:
            messagebox.showerror(
                title="Unknown Side",
                message="This is not an available side of the body.")
            raise KeyError

        if sex == "male":
            self.mesh_name = "Male " + self.mesh_name
        elif sex == "female":
            self.mesh_name = "Female " + self.mesh_name
        else:
            messagebox.showerror(
                title="Unknown Sex",
                message="This sex was not modeled.")
            raise KeyError

        (self.distance_solver, self.path_solver,
         self.uv_array, self.lookup_data, self.imported_data) =\
            load_mesh(self.mesh_name)
        self.uv_kdtree = KDTree(self.uv_array)
        # Set the array of starting and ending verticies to empty
        self.path_verticies = None

    def analyze_data(self, data: np.ndarray) -> None:
        '''
        Loads in data and analyzes it

        Combines the load and compute steps for ease of use

        Parameters
        ----------
        data : np.ndarray
            The input data from the centroids
        '''
        self.load_data(data)
        self.found_distances = self.calculate_distances()
        self.found_paths = self.calculate_paths()

    def analyze_data_from_csv(self) -> None:
        '''
        Loads in points to measure between from a file

        Loads in predetermined points to find geodesic distances between from
        a csv file. If there is a missing starting or ending point value, the
        code ommits that row of points from the loaded in data.
        '''
        filename = askopenfilename(initialdir="../Data",
                                   filetypes=[("data files", "*.csv")])
        # Load in data and exclude rows with any missing values
        location_data =\
            pl.read_csv(filename).drop_nulls()[
                ["start x", "start y",
                 "end x", "end y"]].to_numpy()
        # calculate the geodesic information
        self.analyze_data(location_data)

    def calculate_distances(self) -> np.ndarray:
        '''
        Find the distance between the starting and ending points

        Finds the distances between the two points entered to all other points
        and adds it to the visualization.

        Returns
        -------
        path_distances : np.ndarray
            The found geodesic distances in order of the input data

        Raises
        ------
        ValueError
            If you have not given starting or ending points

        Notes
        -----
        The solvers here uses finds the geodesic distance, i.e. the
        shortest distance between any two points that goes across the mesh.
        This is done using the Heat Method [*]_.

        References
        ----------
        .. [*] Keenan Crane, Clarisse Weischedel, and Max Wardetzky. 2013.
           Geodesics in heat: A new approach to computing distance based on
           heat flow. ACM Trans. Graph. 32, 5, Article 152 (September 2013),
           11 pages https://doi.org/10.1145/2516971.2516977
        '''
        if self.start_x_location is None or self.start_y_location is None:
            print("You have not set any starting points.")
            raise ValueError
        if self.end_x_location is None or self.end_y_location is None:
            print("You have not set a ending point.")
            raise ValueError
        # Find the image dimensions
        img = cv2.imread('../Media/' + self.drawing_name, 1)
        image_x_size = img.shape[1]
        image_y_size = img.shape[0]

        # iterate through each location set
        # Find the UV location of the start and end points
        start_centroids =\
            np.stack((self.start_x_location, self.start_y_location), axis=-1)
        end_centroids =\
            np.stack((self.end_x_location, self.end_y_location), axis=-1)
        start_vertex_ids, end_vertex_ids =\
            self.uv_to_vertex(
                start_centroids, end_centroids, image_x_size, image_y_size)
        with progressbar.ProgressBar(max_value=len(self.start_x_location),
                                     widgets=[Percentage(), " ",
                                              GranularBar(), " ", ETA()]
                                     ) as bar:
            verticies = []
            path_distances = []
            mesh_vertices =\
                np.repeat(
                    self.imported_data["mesh_verticies"][np.newaxis, ...],
                    len(start_vertex_ids), axis=0)
            mesh_faces = np.repeat(
                self.imported_data["mesh_faces"][np.newaxis, ...],
                len(start_vertex_ids), axis=0)
            t1 = time.perf_counter(), time.process_time()
            path_distances =\
                self.p.map(solve,
                           mesh_vertices,
                           mesh_faces,
                           start_vertex_ids,
                           end_vertex_ids)
            t2 = time.perf_counter(), time.process_time()
            print(f" Real time: {t2[0] - t1[0]:.2f} seconds")
            print(f" CPU time: {t2[1] - t1[1]:.2f} seconds")
            for i, ind in enumerate(start_vertex_ids):
                # Find distance from the each start point to all others
                # dist = self.distance_solver.compute_distance(ind)
                # path_distances.append(dist[end_vertex_ids[i]])
                verticies.append([start_vertex_ids[i], end_vertex_ids[i]])
                bar.update(i)

        self.path_verticies = np.array(verticies)
        path_distances = np.array(path_distances)
        print("Distance Calculation Finished")
        return path_distances

    def calculate_paths(self) -> Dict[str, np.ndarray]:
        '''
        Finds the path between the start and end vertex

        The path is found using edge flips until the start vertex connects to
        the end vertex

        Returns
        -------
        data_dict : Dict(str, np.ndarray)/
            The found paths between the input data

        Raises
        ------
        ValueError
            If you did not find starting and ending verticies

        Notes
        -----
        A solver for the path between these two points uses edge flips to show
        the path on the mesh [*]_. Note that this path may not be the
        shortest path, just a demonstration.

        References
        ----------
        .. [*] Nicholas Sharp and Keenan Crane. 2020. You can find
           geodesic paths in triangle meshes by just flipping edges. ACM
           Trans. Graph. 39, 6, Article 249 (December 2020), 15 pages.
           https://doi.org/10.1145/3414685.3417839
        '''
        if self.path_verticies is None:
            messagebox.showerror(
                title="Verticies not Found",
                message="You have not found the starting and ending verticies")
            raise ValueError

        # Find the path
        for i, vertex_set in enumerate(self.path_verticies):
            path_points = self.path_solver.find_geodesic_path_poly(
                vertex_set)
            if i == 0:
                data_dict = {f"calculated_path_{str(i + 1)}": path_points}
            else:
                data_dict[f"calculated_path_{str(i + 1)}"] = path_points

        print("Path Calculation Finished")
        return data_dict

    def load_data(self, data: np.ndarray) -> None:
        '''
        Takes an Nx4 numpy array and converts it to start and end points

        These points are used for finding geodesic distances and paths

        Parameters
        ----------
        data : ndarray
            The input data from the centroids

        Raises
        ------
        TypeError
            If the data is not a numpy array, this function will not work
        '''
        if not isinstance(data, np.ndarray):
            raise TypeError

        self.start_x_location = data[:, 0]
        self.start_y_location = data[:, 1]
        self.end_x_location = data[:, 2]
        self.end_y_location = data[:, 3]

    def uv_to_vertex(self, start_centroid: np.ndarray,
                     end_centroid: np.ndarray,
                     image_x_size: int, image_y_size: int) -> int:
        '''
        Converts location drawing pixel value to 3D vertex location

        Takes the location drawing's centroid pixel location and converts it
        to a 3D vertex on the mesh by finding the closest UV value to the pixel

        Parameters
        ----------
        centroid : np.ndarray
            The x and y pixel values of a start and end location centroid
        image_x_size : int
            The x dimension of the location drawing image in pixels
        image_y_size : int
            The y dimension of the location drawing image in pixels

        Returns
        -------
        nearest_vertex_id : int
            The row numbers of the closest vertex to the start and end
            2D centroids
        '''
        centroid_uv_location = np.zeros((len(start_centroid[:, 0]), 2))
        centroid_uv_location[:, 0] = start_centroid[:, 0] / image_x_size
        centroid_uv_location[:, 1] = 1 - (start_centroid[:, 1] / image_y_size)
        _, nearest_uv_ids = self.uv_kdtree.query(centroid_uv_location)
        start_vertex_ids = []
        for uv_id in nearest_uv_ids:
            start_vertex_id = self.lookup_data.filter(
                pl.col("uv").is_in(uv_id)
                )["vertex"][0]
            start_vertex_ids.append(start_vertex_id)
        centroid_uv_location = np.zeros((len(end_centroid[:, 0]), 2))
        centroid_uv_location[:, 0] = end_centroid[:, 0] / image_x_size
        centroid_uv_location[:, 1] = 1 - (end_centroid[:, 1] / image_y_size)
        _, nearest_uv_ids = self.uv_kdtree.query(centroid_uv_location)
        end_vertex_ids = []
        for uv_id in nearest_uv_ids:
            end_vertex_id = self.lookup_data.filter(
                pl.col("uv").is_in(uv_id)
                )["vertex"][0]
            end_vertex_ids.append(end_vertex_id)

        return start_vertex_ids, end_vertex_ids


if __name__ == "__main__":
    path_finder = GeodesicPath()
    path_finder.analyze_data_from_csv()
    print(path_finder.found_distances)
