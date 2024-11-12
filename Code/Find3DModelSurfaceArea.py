# get3DModelSurfaceArea.py

# Original script by Nabeel Chowdhury 8/7/2023
# Last edit: Sedona Cady 5/3/2024

# Requirements
'''
    Python 3.9.0 - potpourri3d and polyscope do
    not work with newer versions of python
    customtkinter version: 5.1.2 dependency check for customtkinter
    jupyterlab version: 3.5.3 dependency check for jupyterlab
    numpy version: 1.24.2 dependency check for numpy
    opencv-python version: 4.7.0.72 dependency check for opencv
    pandas version: 1.5.3 dependency check for pandas
    polyscope version: 1.3.1 dependency check for polyscope
    potpourri3d version: 0.0.8 dependency check for potpourri3d
    pynput version: 1.7.6 dependency check for pynput
    scipy version: 1.10.1 dependency check for scipy
'''

# Load in libraries
import numpy as np
import matplotlib.path as mpltPath
import pandas as pd
import cv2
from tkinter import filedialog
import os
from scipy.spatial import KDTree
import scipy.io as sio
import pyvista as pv


# Define Function to Find UV vertex indicies
def find_uv_index_kdtree(border_points: pd.DataFrame, image_x_size: int,
                         image_y_size: int) -> int:
    '''
    Converts one location drawing border pixel to a UV value
    Takes a location drawing pixel location and converts it
    to the index the closest UV value to the pixel using a KD Tree
    Parameters
    ----------
    border_points: np.ndarray
    The x and y pixel values of a border point
    image_x_size: int
    The x dimension of the location drawing image in pixels
    image_y_size: int
    The y dimension of the location drawing image in pixels
    Returns
    -------
    indicies: int
    The row numbers of the closest uv to the 2D border point list
    '''
    data_array = border_points.to_numpy(dtype=float)
    data_array[:, 0] = data_array[:, 0] / image_x_size
    data_array[:, 1] = 1 - (data_array[:, 1] / image_y_size)
    _, indicies = KDTree(uv_array).query(data_array)
    return indicies


# Load in border(s) from csv file(s) (Sprompt user)
filepath = (filedialog.askopenfilenames(
    title="Please select .mat file of UV map border coordinates",
    filetypes=[("MAT Files", "*.mat")]))
mat_data = sio.loadmat(filepath)
mat_data.head()
donotsave = 0

# Load in male arm mesh data
meshdata_left = np.load("male left arm mesh data.npz")
meshdata_right = np.load("male right arm mesh data.npz")

# Initialize surface area and label arrays
SA = np.zeros(len(mat_data))
trialnum = np.tile(0, len(mat_data))

''' Initialize average estimate conversion info
    reference: https://msis.jsc.nasa.gov/sections/section03.htm
    reference from the reference:  AMRL-TR-74-102 Churchill,
    E. Sampling and Data Gathering Strategies for Future USAF
    Anthropometry Webb Associates, Inc. A/F Aerospace Medical Res 2-76 '''
hand_length_avg_male = 19.3  # cm, tip of D3 to thenar eminance-wrist junction
hand_length_blender = 1.34604  # arbitrary units (m)
ratio_hand_length = hand_length_avg_male/hand_length_blender  # cm/au
ratio_hand_area = ratio_hand_length**2  # cm^2/au^2

for i in range(0, len(mat_data)):

    # Get boundary data
    data = mat_data[i].Boundaries
    trialnum[i] = i

    # Get mesh data and image data
    if mat_data[i].AmputationSite == "Right hand":
        imported_data = meshdata_right
        img = (cv2.imread(
            r'D:\GitHub\Geodesic-Path-Finder\Media\right arm.png', 1))
    else:
        imported_data = meshdata_left
        img = (cv2.imread(
            r'D:\GitHub\Geodesic-Path-Finder\Media\left arm.png', 1))
    mesh_verticies = imported_data["mesh_verticies"]
    mesh_faces = imported_data["mesh_faces"]

    # Get image dimensions
    image_x_size = img.shape[1]
    image_y_size = img.shape[0]

    # Load in uv data
    uv_array = imported_data["uv_array"]

    # import the face data
    face_data = (pd.DataFrame(
        imported_data["face_data"], columns=["vertex", "uv", "normal"]))

    # Convert the boundary pixel values to their corresponding uv indices
    boundary_uv_array = []
    boundary_uv_array = find_uv_index_kdtree(data, image_x_size, image_y_size)

    # Delete duplicate UV array indices
    cleaned_boundary_uv_array = []
    [cleaned_boundary_uv_array.append(x) for x in boundary_uv_array if (
        x not in cleaned_boundary_uv_array)]

    # Check the original boundary vs the UV boundaries
    path = uv_array[cleaned_boundary_uv_array]

    # Find the UV points inside the UV boundary
    boundary = mpltPath.Path(path)
    inside_boundary = boundary.contains_points(uv_array)

    # Triangluate the Found Mesh
    # This is so we can apply the same triangulation to the 3D mesh

    inside_boundary_ids = [i for i, x in enumerate(inside_boundary) if x]
    combined_uv_array = cleaned_boundary_uv_array.copy()
    combined_uv_array.extend(inside_boundary_ids)

    combined_uv_array_unique = []
    [combined_uv_array_unique.append(x) for x in combined_uv_array if (
        x not in combined_uv_array_unique)]

    combined_uv_array = np.array(combined_uv_array_unique)
    location_uvs = np.array(uv_array[combined_uv_array])

    # Translate the UV Triangulation to the 3D Mesh
    # Sort the UV Indicies
    indicies_of_sorted_indicies = np.argsort(combined_uv_array)
    sorted_indicies = combined_uv_array[indicies_of_sorted_indicies]

    # Make a look up table of one for one UV to Vertex Points
    face_data_reduced = face_data[["vertex", "uv"]].drop_duplicates()
    face_data_reduced = face_data_reduced.sort_values(by=['uv'])

    # Find all Verticies at Once and Reorganize the Array Based on the Sort

    face_data_reduced[face_data_reduced['uv'].isin(combined_uv_array)]

    sorted_vertex_ids =\
        face_data_reduced[face_data_reduced['uv'].
                          isin(combined_uv_array)]["vertex"].to_numpy()

    vertex_ids = np.empty((len(combined_uv_array))).astype(int)

    for index, value in enumerate(indicies_of_sorted_indicies):
        vertex_ids[value] = sorted_vertex_ids[index]

    # Get the Vertex Positions using the Indicies
    location_surface = np.array(mesh_verticies[vertex_ids])

    # Find the Surface Area of the 3D Location Drawing
    # Convert the mesh into a point cloud
    cloud = pv.PolyData(location_surface)

    # Triangulate the 3D Mesh with short triangles
    volume = cloud.delaunay_2d(alpha=0.05)
    shell = volume.extract_geometry()

    # Scale to cm based on average male size
    norm_area = shell.area*ratio_hand_area

    SA[i] = norm_area

if donotsave == 0:
    # Create file to write to
    newfilename = input(
        "Please type the filename you would like to save out: ")
    pathname = r'D:\GitHub\HandDatabase'
    fullfilename = os.path.join(pathname, newfilename)
    fid = open(fullfilename, "w")
    SADataFrame = pd.DataFrame({'RowNum': trialnum, 'SA': SA})
    SADataFrame.to_csv(fid, sep=',', index=False, header=True)

# Close file
fid.close()
