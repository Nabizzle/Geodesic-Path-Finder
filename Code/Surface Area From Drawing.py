
# Load in Libraries
from scipy.spatial import Delaunay, KDTree
import numpy as np
import matplotlib.path as mpltPath
import pandas as pd
import cv2
import math


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


def heron_area(p1, p2, p3):
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p1 - p3)
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


# Load in data
data = pd.read_csv("../Data/ExampleBoundaryTable.csv")
data.head()

# Load in Male Right Arm Mesh Data
imported_data =\
    np.load("../Data/male right arm mesh data.npz")
mesh_verticies = imported_data["mesh_verticies"]
mesh_faces = imported_data["mesh_faces"]

# Load in uv data
uv_array = imported_data["uv_array"]

# import the face data
face_data = pd.DataFrame(imported_data["face_data"],
                         columns=["vertex", "uv", "normal"])

# Load in the Male Right Arm and Find the image dimensions
img = cv2.imread('../Media/right arm.png', 1)
image_x_size = img.shape[1]
image_y_size = img.shape[0]

# Convert the Boundary Pixel Values to Their Indicies in the UV Array

boundary_uv_array = find_uv_index_kdtree(data, image_x_size, image_y_size)

# Delete Duplicate UV Array Indicies

cleaned_boundary_uv_array = []
[cleaned_boundary_uv_array.append(x) for x in boundary_uv_array
 if x not in cleaned_boundary_uv_array]

path = uv_array[cleaned_boundary_uv_array]

# Find the UV Points Inside the UV Boundary
boundary = mpltPath.Path(path)
inside_boundary = boundary.contains_points(uv_array)
# Triangluate the Found Mesh
# This is so we can apply the same triangulation to the 3D mesh

location_uvs = np.array(uv_array[cleaned_boundary_uv_array])
location_uvs = np.concatenate((location_uvs, uv_array[inside_boundary]))
triangulated_uvs = Delaunay(location_uvs)

# Eliminate Triangles that Only Connect to Border Points
highest_boundary_index = len(cleaned_boundary_uv_array)
triangles = np.array(triangulated_uvs.simplices)
reduced_triangles = []
for row in triangles:
    if not np.all(row <= highest_boundary_index):
        reduced_triangles.append(row)
reduced_triangles = np.array(reduced_triangles)

# Translate the UV Triangulation to the 3D Mesh
# Find the Boundary Verticies
boundary_vertex_ids = []
for index in cleaned_boundary_uv_array:
    boundary_vertex_ids.append(int(face_data.loc[face_data["uv"] == index]
                                   ["vertex"].values[0]))

# Find Boundary Vertex Indicies
# Get indicies of true values in inside_boundary
inside_boundary_ids = [i for i, x in enumerate(inside_boundary) if x]
inner_vertex_ids = [int(face_data.loc[face_data["uv"] == index]
                    ["vertex"].values[0]) for index in inside_boundary_ids]

# Get the Vertex Positions using the Indicies
location_surface = np.array(mesh_verticies[boundary_vertex_ids])
location_surface = np.concatenate((location_surface,
                                   mesh_verticies[inner_vertex_ids]))

# Find the Surface Area of the 3D Location Drawing
# Calculate the Area Using Heron's Formula
area = 0
for tri in reduced_triangles:
    p1 = location_surface[tri[0]]
    p2 = location_surface[tri[1]]
    p3 = location_surface[tri[2]]
    area += heron_area(p1, p2, p3)

print(f"The area of the drawn location is {area} scene units")
