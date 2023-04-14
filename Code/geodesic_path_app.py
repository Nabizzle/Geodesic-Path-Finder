import customtkinter as ctk
import cv2
import numpy as np
import pandas as pd
import polyscope as ps
import potpourri3d as pp3d
from tkinter import messagebox
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
    start_x_location: int
        The x pixel value of the starting location drawing centroid
    start_y_location: int
        The y pixel value of the starting location drawing centroid
    end_x_location: int
        The x pixel value of the starting location drawing centroid
    end_y_location: int
        The y pixel value of the starting location drawing centroid
    centroid_entry_frame: CentroidEntry
        Frame containing the centroid entry widgets
    set_start_location_button: CTkButton
        Button to get the start centroid locations and set them in the correct
        attributes
    set_end_location_button: CTkButton
        Button to get the start centroid locations and set them in the correct
        attributes

    Methods
    -------
    __init__()
        Creates the GUI elements for the app
    set_mesh_name(choice)
        Set the mesh name based on the dropdown choice
    load_mesh()
        Load the selected mesh and create the distance solver

    See Also
    --------
    CentroidEntry
    '''
    PADX = 10
    PADY = 10
    WIDGET_FONT = ("Inter", 16)

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
        self.mesh_name = "Right Arm UV Mapped.obj"
        self.drawing_name = "right arm.png"
        self.text_mesh_data_name = "Right Arm UV Mapped as Text.txt"
        default_arm = ctk.StringVar(value="Right Arm")
        self.mesh_selection_dropdown = ctk.CTkOptionMenu(
            self, values=["Right Arm", "Left Arm"], command=self.set_mesh_name,
            variable=default_arm, font=self.WIDGET_FONT)
        self.mesh_selection_dropdown.grid(row=0, column=0, padx=self.PADX,
                                          pady=self.PADY)

        # Button to load in the mesh
        self.mesh_verticies, self.mesh_faces = None, None
        self.polyscope_mesh = None
        self.solver = None
        self.load_mesh_button = ctk.CTkButton(
            self, text="Load Mesh", command=self.load_mesh,
            font=self.WIDGET_FONT)
        self.load_mesh_button.grid(row=0, column=1, padx=self.PADX,
                                   pady=self.PADY)

        # Centroid entry frame
        self.start_x_location, self.start_y_location = -1, -1
        self.end_x_location, self.end_y_location = -1, -1
        self.centroid_entry_frame = CentroidEntry(self)
        self.centroid_entry_frame.grid(
            row=1, column=0, rowspan=2, padx=self.PADX,
            pady=self.PADY)
        self.set_start_location_button = ctk.CTkButton(
            self, text="Save Location", command=self.save_start_location,
            font=self.WIDGET_FONT)
        self.set_start_location_button.grid(
            row=1, column=1, padx=self.PADX, pady=self.PADY)
        self.set_end_location_button = ctk.CTkButton(
            self, text="Save Location", command=self.save_end_location,
            font=self.WIDGET_FONT)
        self.set_end_location_button.grid(
            row=2, column=1, padx=self.PADX, pady=self.PADY)

        # Manual centroid point selection
        self.click_start_point_button = ctk.CTkButton(
            self, text="Select Start Point",
            command=self.manual_start_selection, font=self.WIDGET_FONT)
        self.click_start_point_button.grid(row=3, column=0,
                                           padx=self.PADX, pady=self.PADY)
        self.click_end_point_button = ctk.CTkButton(
            self, text="Select End Point",
            command=self.manual_start_selection, font=self.WIDGET_FONT)
        self.click_end_point_button.grid(row=3, column=1,
                                         padx=self.PADX, pady=self.PADY)

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
        messagebox.showinfo(
            title="Load Completed",
            message="Mesh loading finished")

    def save_start_location(self) -> None:
        ''' Save the start centroid locations '''
        self.start_x_location, self.start_y_location =\
            self.centroid_entry_frame.get_start_locations()

    def save_end_location(self) -> None:
        ''' Save the end centroid locations '''
        self.end_x_location, self.end_y_location =\
            self.centroid_entry_frame.get_end_locations()

    def manual_start_selection(self) -> None:
        self.start_x_location, self.start_y_location =\
            self.image_point_selection()

    def manual_end_selection(self) -> None:
        self.start_x_location, self.start_y_location =\
            self.image_point_selection()

    def image_point_selection(self) -> Tuple:
        img = cv2.imread('../Media/' + self.drawing_name, 1)
        image_x_size = img.shape[1]
        image_y_size = img.shape[0]
        scale_factor = 3
        img = cv2.resize(
            img, (int(image_x_size / scale_factor),
                  int(image_y_size / scale_factor)))
        cv2.imshow('image', img)
        click_x, click_y = -1, -1
        cv2.setMouseCallback('image', self.click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return (click_x, click_y)

    def click_event(self, event: int, x: float, y: float, flags,
                    params) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            global click_x, click_y
            click_x, click_y = x, y


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
            The x and y locations of the starting point as integers

        See Also
        --------
        save_end_location
        '''
        start_x_location = int(np.round(float(self.start_point_x.get())))
        start_y_location = int(np.round(float(self.start_point_y.get())))
        return (start_x_location, start_y_location)

    def get_end_locations(self) -> Tuple:
        '''
       Return the x and y locations entered into the end row

        Returns
        -------
        Tuple
            The x and y locations of the ending point as integers

        See Also
        --------
        save_start_location
        '''
        end_x_location = int(np.round(float(self.end_point_x.get())))
        end_y_location = int(np.round(float(self.end_point_y.get())))
        return (end_x_location, end_y_location)


if __name__ == "__main__":
    # Create app and run it
    app = GeodesicPathApp()
    app.mainloop()
