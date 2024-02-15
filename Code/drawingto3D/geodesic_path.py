from tkinter import messagebox
import numpy as np
import polars as pl
import cv2
from typing import Dict
from tkinter.filedialog import askopenfilename
from data_manager import load_mesh


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
    def __init__(self, sex: str = "male", side: str = "right") -> None:
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
         self.uv_array, self.lookup_data) = load_mesh(self.mesh_name)
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
        if isinstance(self.start_x_location, np.ndarray):
            self.path_verticies = np.array([])
            for i in range(len(self.start_x_location)):
                # Find the UV location of the start and end points
                start_vertex_id = self.uv_to_vertex(
                    self.start_x_location[i], self.start_y_location[i],
                    image_x_size, image_y_size)
                end_vertex_id = self.uv_to_vertex(
                    self.end_x_location[i], self.end_y_location[i],
                    image_x_size, image_y_size)

                # Find distance from the start and end points to all others
                dist = self.distance_solver.compute_distance(start_vertex_id)
                if i == 0:
                    self.path_verticies = np.array(
                        [[start_vertex_id, end_vertex_id]])
                    path_distances = np.array([dist[end_vertex_id]])
                else:
                    self.path_verticies = np.append(
                        self.path_verticies,
                        [[start_vertex_id, end_vertex_id]], 0)
                    path_distances = np.append(
                        path_distances, [dist[end_vertex_id]], 0)

        else:
            # Find the UV location of the start and end points
            start_vertex_id = self.uv_to_vertex(
                self.start_x_location, self.start_y_location, image_x_size,
                image_y_size)
            end_vertex_id = self.uv_to_vertex(
                self.end_x_location, self.end_y_location, image_x_size,
                image_y_size)

            # Find distance from the start and end points to all others
            self.path_verticies = np.array([[start_vertex_id, end_vertex_id]])
            dist = self.distance_solver.compute_distance(start_vertex_id)
            path_distances = np.array([dist[end_vertex_id]])

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

    def uv_to_vertex(self, centroid_x: float, centroid_y: float,
                     image_x_size: int, image_y_size: int) -> int:
        '''
        Converts location drawing pixel value to 3D vertex location

        Takes the location drawing's centroid pixel location and converts it
        to a 3D vertex on the mesh by finding the closest UV value to the pixel

        Parameters
        ----------
        centroid_x : float
            The x pixel value of a location centroid
        centroid_y : float
            The y pixel value of a location centroid
        image_x_size : int
            The x dimension of the location drawing image in pixels
        image_y_size : int
            The y dimension of the location drawing image in pixels

        Returns
        -------
        nearest_vertex_id : int
            The row number of the closest vertex to the 2D centroid
        '''
        normalized_x = centroid_x / image_x_size
        normalized_y = centroid_y / image_y_size
        centroid_uv_location = np.array([normalized_x, 1 - normalized_y])
        distances_to_uvs = np.linalg.norm(
            self.uv_array - centroid_uv_location, axis=1)
        nearest_uv_id = distances_to_uvs.argsort()[0]
        nearest_vertex_id = int(
            self.lookup_data.filter(
                pl.col("uv") == nearest_uv_id
            )["vertex"][0]) - 1
        return nearest_vertex_id


if __name__ == "__main__":
    path_finder = GeodesicPath()
    path_finder.analyze_data_from_csv()
    print(path_finder.found_distances)
