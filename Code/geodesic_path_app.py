import customtkinter as ctk
import cv2
import numpy as np
import pandas as pd
import polyscope as ps
import potpourri3d as pp3d
from pynput.keyboard import Key, Controller
from tkinter import messagebox
from tkinter.filedialog import askopenfilename
from typing import Tuple

ctk.set_appearance_mode("System")
ctk.set_default_color_theme("dark-blue")


class GeodesicPathApp(ctk.CTk):
    '''
    App for finding the geodesic distance between two points on a 3D model

    The app finds the distance between two points on a 3D mesh given their
    location on a 2D drawing. These 2D locations are translated to a 3D
    location by mapping the 3D mesh faces in x, y, and z to u and v
    coordinates on the 2D drawing. The app can output a visualization of the
    distances as well and show the path between the locations.

    Attributes
    ----------
    PADX: int
        The number of pixels to pad on the left and right sides of the GUI
        elements
    PADY: int
        The number of pixels to pad on the top and bottom sides of the GUI
        elements
    WIDGET_FONT: tuple (str, int)
        The font for the GUI widgets called as the font name and font size
    LABEL_FONT: tuple (str, int)
        The font for the GUI labels called as the font name and font size
    mesh_options_label: CTkLabel
        The label for the mesh control row
    mesh_name: str
        The name of the mesh to find the geodesic distance and path on
    drawing_name: str
        The name of the 2D location drawing template
    text_mesh_data_name: str
        The name of the mesh data text file
    mesh_selection_dropdown: CTkOptionMenu
        dropdown to select the name of the mesh to use
    mesh_verticies: ndarray
        numpy array of all verticies in the selected mesh as x, y, and z
        coordinates
    mesh_faces: ndarray
        numpy array of all faces in the selected mesh. For the code to work,
        the faces need to be trianglulated resulting in an array of dimensions
        Nx3.
    polyscope_mesh: SurfaceMesh
        The mesh in the visualization in polyscope
    solver: MeshHeatMethodDistanceSolver
        MeshHeatMethodDistanceSolver object for finding geodesic distances
        using the heat method
    load_mesh_button: CTkButton
        Button to load in the mesh and create the distance solver
    vertex_data: DataFrame
        A table of all of the verticies in the mesh
    uv_data: DataFrame
        A table of all of the uv data of the mesh
    start_x_location: float or ndarray
        The x pixel value of the starting location drawing centroid
    start_y_location: float or ndarray
        The y pixel value of the starting location drawing centroid
    end_x_location: float or ndarray
        The x pixel value of the starting location drawing centroid
    end_y_location: float or ndarray
        The y pixel value of the starting location drawing centroid
    centroid_entry_frame: CentroidEntry
        Frame containing the centroid entry widgets
    set_start_location_button: CTkButton
        Button to get the start centroid locations and set them in the correct
        attributes
    set_end_location_button: CTkButton
        Button to get the start centroid locations and set them in the correct
        attributes
    click_start_point_button: CTkButton
        Button to manually click the starting locaiton of a path
    click_end_point_button: CTkButton
        Button to manually click the ending locaiton of a path
    click_x: float
        The x location the user clicked on on the 2D image
    click_y: float
        The y location the user clicked on on the 2D image
    load_points_button: CTkButton
        Button for loading in starting and ending points on the location
        drawing from a csv file
    path_verticies: ndarray
        An array of the start and end vertex numbers for the location drawing
        centriods
    path_distances: ndarray
        An array of the distances from one set of verticies to all other
        verticies as organized as by the vertex numbers.

    Methods
    -------
    __init__()
        Creates the GUI elements for the app
    set_mesh_name(choice)
        Set the mesh name based on the dropdown choice
    load_mesh()
        Load the selected mesh and create the distance solver
    save_start_location()
        Get the start point location from the centroid entry fields
    save_end_location()
        Get the end point location from the centroid entry fields
    manual_start_selection()
        Open a location drawing image to find a start centroid value on
    manual_end_selection
        Open a location drawing image to find an end centroid value on
    image_point_selection()
        Handles opening an image for clicking on
    click_event(event, x, y, flags, params)
        mouse event callback for getting a clicked on point
    calculate_distances()
        Find the distance between the starting and ending points
    load_points()
        Loads in points to measure between from a file
    uv_to_vertex(centroid_x, centroid_y, image_x_size, image_y_size)
        Converts location drawing pixel value to 3D vertex location
    calculate_path
        Finds the path between the start and end vertex
    show_visualization
        Shows the mesh in polyscope

    See Also
    --------
    CentroidEntry
    '''
    PADX = 10
    PADY = 10
    WIDGET_FONT = ("Inter", 16)
    LABEL_FONT = ("Inter", 20)

    def __init__(self) -> None:
        '''
        Creates the GUI elements for the app
        '''
        super().__init__()

        ps.init()  # Initialize polyscope for visualization

        # Global app parameters
        self.title("Geodesic Path Finder")
        self.config(padx=self.PADX, pady=self.PADY)
        self.resizable(False, False)

        # Mesh Selection Dropdown Menu
        self.mesh_options_label = ctk.CTkLabel(
            self, text="Select Mesh", font=self.LABEL_FONT)
        self.mesh_options_label.grid(
            row=0, column=0, padx=self.PADX, pady=self.PADY)
        self.mesh_name = "Right Arm UV Mapped.obj"
        self.drawing_name = "right arm.png"
        self.text_mesh_data_name = "Right Arm UV Mapped as Text.txt"
        default_arm = ctk.StringVar(value="Right Arm")
        self.mesh_selection_dropdown = ctk.CTkOptionMenu(
            self, values=["Right Arm", "Left Arm"], command=self.set_mesh_name,
            variable=default_arm, font=self.WIDGET_FONT)
        self.mesh_selection_dropdown.grid(row=0, column=1, padx=self.PADX,
                                          pady=self.PADY)

        # Button to load in the mesh
        self.mesh_verticies, self.mesh_faces = None, None
        self.polyscope_mesh = None
        self.solver = None
        self.load_mesh_button = ctk.CTkButton(
            self, text="Load Mesh", command=self.load_mesh,
            font=self.WIDGET_FONT)
        self.load_mesh_button.grid(row=0, column=2, padx=self.PADX,
                                   pady=self.PADY)

        # Centroid entry frame
        self.start_x_location, self.start_y_location = None, None
        self.end_x_location, self.end_y_location = None, None
        self.centroid_entry_frame = CentroidEntry(self)
        self.centroid_entry_frame.grid(
            row=1, column=0, rowspan=2, columnspan=2, padx=self.PADX,
            pady=self.PADY)
        self.set_start_location_button = ctk.CTkButton(
            self, text="Save Location", command=self.save_start_location,
            font=self.WIDGET_FONT)
        self.set_start_location_button.grid(
            row=1, column=2, padx=self.PADX, pady=self.PADY)
        self.set_end_location_button = ctk.CTkButton(
            self, text="Save Location", command=self.save_end_location,
            font=self.WIDGET_FONT)
        self.set_end_location_button.grid(
            row=2, column=2, padx=self.PADX, pady=self.PADY)

        # Manual centroid point selection
        self.click_start_point_button = ctk.CTkButton(
            self, text="Select Start Point",
            command=self.manual_start_selection, font=self.WIDGET_FONT)
        self.click_start_point_button.grid(
            row=3, column=0, padx=self.PADX, pady=self.PADY)
        self.click_end_point_button = ctk.CTkButton(
            self, text="Select End Point",
            command=self.manual_end_selection, font=self.WIDGET_FONT)
        self.click_end_point_button.grid(
            row=3, column=1, padx=self.PADX, pady=self.PADY)

        # Load in multiple points from file
        self.load_points_button = ctk.CTkButton(
            self, text="Load Points from File", command=self.load_points,
            font=self.WIDGET_FONT)
        self.load_points_button.grid(
            row=3, column=2, padx=self.PADX, pady=self.PADY)

        # Find Distances
        self.path_verticies = None
        self.calculate_distances_button = ctk.CTkButton(
            self, text="Calculate Distances",
            command=self.calculate_distances, font=self.WIDGET_FONT)
        self.calculate_distances_button.grid(
            row=4, column=0, padx=self.PADX, pady=self.PADY)

        # Find Geodesic Path
        self.calculate_path_button = ctk.CTkButton(
            self, text="Calculate Path",
            command=self.calculate_path, font=self.WIDGET_FONT)
        self.calculate_path_button.grid(
            row=4, column=1, padx=self.PADX, pady=self.PADY)

        # Show visualization
        self.show_visualization_button = ctk.CTkButton(
            self, text="Show Mesh",
            command=self.show_visualization, font=self.WIDGET_FONT)
        self.show_visualization_button.grid(
            row=4, column=2, padx=self.PADX, pady=self.PADY)

    def set_mesh_name(self, choice: str) -> None:
        '''
        Set the mesh name based on the dropdown choice

        Raises
        ------
        KeyError
            If the mesh name selected that has not been implimented
        '''
        if choice == "Right Arm":
            self.mesh_name = "Right Arm UV Mapped.obj"
            self.drawing_name = "right arm.png"
            self.text_mesh_data_name = "Right Arm UV Mapped as Text.txt"
        elif choice == "Left Arm":
            self.mesh_name = "Left Arm UV Mapped.obj"
            self.drawing_name = "left arm.png"
            self.text_mesh_data_name = "Left Arm UV Mapped as Text.txt"
        else:
            messagebox.showerror(
                title="Unknown Mesh",
                message="This is not an available mesh name")
            raise KeyError

    def load_mesh(self) -> None:
        '''
        Load the selected mesh and create the distance solver

        Load in the mesh selected from the mesh name dropdown menu and create a
        heat method distance solver for the mesh as a stateful solver so
        repeated comutations are quick.
        '''
        # read in the mesh into the class attributes
        self.mesh_verticies, self.mesh_faces = pp3d.read_mesh(
            "../Models/" + self.mesh_name)

        # Add the mesh to polyscope
        self.polyscope_mesh = ps.register_surface_mesh(
            "mesh", self.mesh_verticies, self.mesh_faces)

        # Create the heat method solver for the mesh
        self.solver = pp3d.MeshHeatMethodDistanceSolver(
            self.mesh_verticies, self.mesh_faces)

        # Load in the text version of the model
        mesh_data = pd.read_csv(
            "../Models/" + self.text_mesh_data_name,
            names=["Type", "Point 1", "Point 2", "Point 3"],
            delim_whitespace=True, dtype=str)
        grouped_mesh_data = mesh_data.groupby(["Type"])

        # Extract out the vertex data
        self.vertex_data = grouped_mesh_data.get_group("v")
        self.vertex_data = self.vertex_data.astype(
            {"Point 1": float, "Point 2": float, "Point 3": float})
        self.vertex_data.drop("Type", axis=1, inplace=True)
        self.vertex_data.rename(
            columns={"Point 1": "x", "Point 2": "y", "Point 3": "z"},
            inplace=True)
        self.vertex_data.reset_index(drop=True, inplace=True)
        self.vertex_data.index += 1

        # Get the UV data
        self.uv_data = grouped_mesh_data.get_group("vt")
        self.uv_data = self.uv_data.astype(
            {"Point 1": float, "Point 2": float, "Point 3": float})
        self.uv_data.drop("Type", axis=1, inplace=True)
        self.uv_data.drop("Point 3", axis=1, inplace=True)
        self.uv_data.rename(columns={"Point 1": "x", "Point 2": "y"},
                            inplace=True)
        self.uv_data.reset_index(drop=True, inplace=True)
        self.uv_data.index += 1

        # Get the face data
        self.face_data = grouped_mesh_data.get_group("f")
        split_face_data_1 = pd.DataFrame()
        split_face_data_1[["vertex", "uv", "normal"]] =\
            self.face_data["Point 1"].str.split("/", expand=True)
        split_face_data_2 = pd.DataFrame()
        split_face_data_2[["vertex", "uv", "normal"]] =\
            self.face_data["Point 2"].str.split("/", expand=True)
        split_face_data_3 = pd.DataFrame()
        split_face_data_3[["vertex", "uv", "normal"]] =\
            self.face_data["Point 3"].str.split("/", expand=True)
        split_face_data = pd.concat(
            [split_face_data_1, split_face_data_2, split_face_data_3])
        split_face_data = split_face_data.astype(
            {"vertex": float, "uv": float, "normal": float})
        self.face_data = split_face_data.reset_index(drop=True)
        self.face_data.index += 1

        messagebox.showinfo(
            title="Load Completed", message="Mesh loading finished")

    def save_start_location(self) -> None:
        ''' Save the start centroid locations '''
        self.start_x_location, self.start_y_location =\
            self.centroid_entry_frame.get_start_locations()

    def save_end_location(self) -> None:
        ''' Save the end centroid locations '''
        self.end_x_location, self.end_y_location =\
            self.centroid_entry_frame.get_end_locations()

    def manual_start_selection(self) -> None:
        '''Open a location drawing image to find a start centroid value on'''
        self.start_x_location, self.start_y_location =\
            self.image_point_selection()
        print(self.start_x_location, " ", self.start_y_location)

    def manual_end_selection(self) -> None:
        '''Open a location drawing image to find an end centroid value on'''
        self.end_x_location, self.end_y_location =\
            self.image_point_selection()
        print(self.end_x_location, " ", self.end_y_location)

    def image_point_selection(self) -> Tuple:
        '''Handles opening an image for clicking on'''
        img = cv2.imread('../Media/' + self.drawing_name, 1)
        image_x_size = img.shape[1]
        image_y_size = img.shape[0]
        scale_factor = 3
        img = cv2.resize(
            img, (int(image_x_size / scale_factor),
                  int(image_y_size / scale_factor)))
        cv2.imshow('image', img)
        self.click_x, self.click_y = -1, -1
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return (self.click_x * scale_factor, self.click_y * scale_factor)

    def click_event(self, event: int, x: float, y: float, flags: str,
                    params) -> None:
        '''
        mouse event callback for getting a clicked on point

        Parameters
        ----------
        event: int
            The OpenCV event id. Code only looks for the left mouse button
            going down
        x: int
            The x location of the mouse on an image
        y: int
            The y location of the mouse on an image
        flags: str
            Any event flags (unused)
        params: List
            Any extra input paramters (unused)
        '''
        if event == cv2.EVENT_LBUTTONDOWN:
            self.click_x, self.click_y = x, y
            keyboard = Controller()
            keyboard.press(Key.enter)
            keyboard.release(Key.enter)

    def load_points(self) -> None:
        '''
        Loads in points to measure between from a file

        Loads in predetermined points to find geodesic distances between from
        a csv file and splits them up into starting and ending point pairs.
        If there is a missing starting or ending point value, the code lets
        the user know and ommits that row of points from the loaded in data.
        '''
        filename = askopenfilename()
        # Load in data and exclude rows with any missing values
        location_data = pd.read_csv(filename).dropna()
        # Split up data into the right class attributes
        self.start_x_location = location_data["start x"].to_numpy()
        self.start_y_location = location_data["start y"].to_numpy()
        self.end_x_location = location_data["end x"].to_numpy()
        self.end_y_location = location_data["end y"].to_numpy()

    def calculate_distances(self) -> None:
        '''
        Find the distance between the starting and ending points

        Finds the distances between the two points entered to all other points
        and adds it to the visualization.
        '''
        if self.start_x_location is None or self.start_y_location is None:
            messagebox.showerror(
                title="Starting Centroid not Entered",
                message="You have not set a starting point.")
            raise ValueError
        if self.end_x_location is None or self.end_y_location is None:
            messagebox.showerror(
                title="Ending Centroid not Entered",
                message="You have not set a ending point.")
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

                # Find distance from the start and end points to all other
                dist = self.solver.compute_distance_multisource(
                    [start_vertex_id, end_vertex_id])
                if i == 0:
                    self.path_verticies = np.array(
                        [[start_vertex_id, end_vertex_id]])
                    self.path_distances = np.array([dist])
                else:
                    self.path_verticies = np.append(
                        self.path_verticies,
                        [[start_vertex_id, end_vertex_id]], 0)
                    self.path_distances = np.append(
                        self.path_distances, [dist], 0)

        else:
            # Find the UV location of the start and end points
            start_vertex_id = self.uv_to_vertex(
                self.start_x_location, self.start_y_location, image_x_size,
                image_y_size)
            end_vertex_id = self.uv_to_vertex(
                self.end_x_location, self.end_y_location, image_x_size,
                image_y_size)

            # Find distance from the start and end points to all other
            self.path_verticies = np.array([[start_vertex_id, end_vertex_id]])
            dist = self.solver.compute_distance_multisource(
                [start_vertex_id, end_vertex_id])
            self.path_distances = np.array([dist])

            # Add distances to the visualization
            self.polyscope_mesh.add_distance_quantity(
                "dist", dist, enabled=True, stripe_size=0.01)

        messagebox.showinfo(
            title="Calculation Complete",
            message="Distance Calculation Finished")

    def uv_to_vertex(self, centroid_x: float, centroid_y: float,
                     image_x_size: int, image_y_size: int) -> int:
        '''
        Converts location drawing pixel value to 3D vertex location

        Takes the location drawing's centroid pixel location and converts it
        to a 3D vertex on the mesh by finding the closest UV value to the pixel

        Parameters
        ----------
        centroid_x: float
            The x pixel value of a location centroid
        centroid_y: float
            The y pixel value of a location centroid
        image_x_size: int
            The x dimension of the location drawing image in pixels
        image_y_size: int
            The y dimension of the location drawing image in pixels

        Returns
        -------
        nearest_vertex_id: int
            The row number of the closest vertex to the 2D centroid
        '''
        normalized_x = centroid_x / image_x_size
        normalized_y = centroid_y / image_y_size
        centroid_uv_location = np.array([normalized_x, 1 - normalized_y])
        uv_array = self.uv_data.values  # convert the uv data to a numpy array
        distances_to_uvs = np.linalg.norm(
            uv_array - centroid_uv_location, axis=1)
        nearest_uv_id = distances_to_uvs.argsort()[0]
        nearest_vertex_id = int(
            self.face_data.loc[self.face_data["uv"] ==
                               nearest_uv_id]["vertex"].values[0])
        return nearest_vertex_id

    def calculate_path(self) -> None:
        '''
        Finds the path between the start and end vertex

        The path is found using edge flips until the start vertex connects to
        the end vertex
        '''
        if self.mesh_verticies is None or self.mesh_faces is None:
            messagebox.showerror(
                title="Mesh not Loaded",
                message="You have not loaded in a mesh.")
            raise ValueError
        # create the path solver
        path_solver = pp3d.EdgeFlipGeodesicSolver(
            self.mesh_verticies, self.mesh_faces)

        if self.path_verticies is None:
            messagebox.showerror(
                title="Verticies not Found",
                message="You have not found the starting and ending verticies")
            raise ValueError
        # Find the path
        for i, vertex_set in enumerate(self.path_verticies):
            path_points = path_solver.find_geodesic_path_poly(
                vertex_set)
            # Add the path to the visualization
            ps.register_curve_network(
                "Geodesic Path " + str(i), path_points, edges='line',
                radius=0.002)

        messagebox.showinfo(
            title="Calculation Complete",
            message="Path Calculation Finished")

    def show_visualization(self):
        '''Shows the mesh in polyscope'''
        ps.show()


class CentroidEntry(ctk.CTkFrame):
    '''
    Container for entering the start and end centroid values of a location

    Allows the user to enter pixel values for the centroids of location
    drawings on the 2D drawings. These locations are used for finding the
    geodesic distances and paths.

    Attributes
    ----------
    PADX: int
        The number of pixels to pad on the left and right sides of the GUI
        elements
    PADY: int
        The number of pixels to pad on the top and bottom sides of the GUI
        elements
    LABEL_FONT: tuple (str, int)
        The font for the GUI labels called as the font name and font size
    WIDGET_FONT: tuple (str, int)
        The font for the GUI widgets called as the font name and font size
    start_point_text: CTkLabel
        Label for the start location of the geodesic distance
    start_point_x: CTkEntry
        Entry box for the x location of a drawing centroid in pixels
    start_point_y: CTkEntry
        Entry box for the y location of a drawing centroid in pixels
    end_point_text: CTkLabel
        Label for the start location of the geodesic distance
    end_point_x: CTkEntry
        Entry box for the x location of a drawing centroid in pixels
    end_point_y: CTkEntry
        Entry box for the y location of a drawing centroid in pixels

    Methods
    -------
    get_start_locations()
        Return the x and y locations entered into the start row
    get_end_locations()
        Return the x and y locations entered into the end row

    See Also
    --------
    GeodesicPathApp
    '''
    PADX = 10
    PADY = 10
    LABEL_FONT = ("Inter", 20)
    WIDGET_FONT = ("Inter", 16)

    def __init__(self, *args, **kwargs) -> None:
        '''
        Setup the elements of the centroid entry boxes
        '''
        super().__init__(*args, **kwargs)
        # Set frame to have same color as background
        self.configure(fg_color="transparent")

        # Start location
        self.start_point_text = ctk.CTkLabel(self, text="Start Point",
                                             font=self.LABEL_FONT)
        self.start_point_text.grid(
            row=0, column=0, padx=self.PADX, pady=self.PADY)
        self.start_point_x = ctk.CTkEntry(
            self, placeholder_text="x value", width=60, font=self.WIDGET_FONT)
        self.start_point_x.grid(
            row=0, column=1, padx=self.PADX, pady=self.PADY)
        self.start_point_y = ctk.CTkEntry(
            self, placeholder_text="y value", width=60, font=self.WIDGET_FONT)
        self.start_point_y.grid(
            row=0, column=2, padx=self.PADX, pady=self.PADY)

        # End location
        self.end_point_text = ctk.CTkLabel(self, text="End Point",
                                           font=self.LABEL_FONT)
        self.end_point_text.grid(
            row=1, column=0, padx=self.PADX, pady=self.PADY)
        self.end_point_x = ctk.CTkEntry(
            self, placeholder_text="x value", width=60, font=self.WIDGET_FONT)
        self.end_point_x.grid(
            row=1, column=1, padx=self.PADX, pady=self.PADY)
        self.end_point_y = ctk.CTkEntry(
            self, placeholder_text="y value", width=60, font=self.WIDGET_FONT)
        self.end_point_y.grid(
            row=1, column=2, padx=self.PADX, pady=self.PADY)

    def get_start_locations(self) -> Tuple:
        '''
        Return the x and y locations entered into the start row

        Returns
        -------
        Tuple
            The x and y locations of the starting point as floats

        See Also
        --------
        save_end_location
        '''
        start_x_location = float(self.start_point_x.get())
        start_y_location = float(self.start_point_y.get())
        return (start_x_location, start_y_location)

    def get_end_locations(self) -> Tuple:
        '''
       Return the x and y locations entered into the end row

        Returns
        -------
        Tuple
            The x and y locations of the ending point as floats

        See Also
        --------
        save_start_location
        '''
        end_x_location = float(self.end_point_x.get())
        end_y_location = float(self.end_point_y.get())
        return (end_x_location, end_y_location)


if __name__ == "__main__":
    # Create app and run it
    app = GeodesicPathApp()
    app.mainloop()
