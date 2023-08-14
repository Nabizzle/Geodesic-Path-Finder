
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


def find_triangle_sides(p1: np.ndarray, p2: np.ndarray,
                        p3: np.ndarray) -> np.ndarray:
    '''
    Finds the area of a triangle usign Heron's formula

    Parameters
    ----------
    p1: np.ndarray
        The first (x, y) point of the triangle
    p2: np.ndarray
        The second (x, y) point of the triangle
    p3: np.ndarray
        The thrid (x, y) point of the triangle

    Returns
    -------
    np.ndarray
        Sides of the triangle [a, b, c]
    '''
    a = np.linalg.norm(p1 - p2)
    b = np.linalg.norm(p2 - p3)
    c = np.linalg.norm(p1 - p3)
    return np.array([[a, b, c]])


def heron_area(sides: np.ndarray) -> float:
    '''
    Finds the area of a triangle usign Heron's formula

    Parameters
    ----------
    p1: np.ndarray
        The first (x, y) point of the triangle
    p2: np.ndarray
        The second (x, y) point of the triangle
    p3: np.ndarray
        The thrid (x, y) point of the triangle

    Returns
    -------
    float
        The area of the triangle
    '''
    a = sides[0]
    b = sides[1]
    c = sides[2]
    s = (a + b + c) / 2
    return math.sqrt(s * (s - a) * (s - b) * (s - c))


def remove_area_outliers(area_data: np.ndarray,
                         deviations: int = 3) -> np.ndarray:
    '''
    Remove outliers using the distance away from the median

    The median is a more robust measure compared to the mean as the mean is
    biased by outliers. The median absolute deviation is also a substitute for
    the standard deviation when using the median.

    Parameters
    ----------
    area_data: np.ndarray
    The areas of all of the calculated triangles
    deviations: int
    The number of deviations away from the median to use as an outlier

    Returns
    -------
    cleaned_data: np.ndarray
    The data with outliers removed
    '''
    distance_from_median = np.abs(area_data - np.median(area_data))
    # find the median absolute deviate (similar to standard deviation)
    median_absolute_deviation = np.median(distance_from_median)
    if median_absolute_deviation > 0:
        deviations_away_from_median =\
            distance_from_median / median_absolute_deviation
    else:
        deviations_away_from_median = np.zeros(len(distance_from_median))

    cleaned_data = area_data[deviations_away_from_median < deviations]
    return cleaned_data


def remove_side_outliers(sides_array: np.ndarray,
                         deviations: int = 3) -> np.ndarray:
    '''
    Remove outliers using the distance away from the median

    The median is a more robust measure compared to the mean as the mean is
    biased by outliers. The median absolute deviation is also a substitute for
    the standard deviation when using the median.

    Parameters
    ----------
    sides_array: np.ndarray
    The side lengths of all of the calculated triangles
    deviations: int
    The number of deviations away from the median to use as an outlier

    Returns
    -------
    cleaned_data: np.ndarray
    The data with outliers removed
    '''
    distance_from_median = np.abs(sides_array - np.median(sides_array))
    median_absolute_deviation = np.median(distance_from_median)
    if median_absolute_deviation > 0:
        deviations_away_from_median =\
            distance_from_median / median_absolute_deviation
    else:
        deviations_away_from_median = np.zeros((len(distance_from_median), 3))
    cleaned_data =\
        sides_array[np.all(deviations_away_from_median < deviations, axis=1)]
    return cleaned_data


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

inside_boundary_ids = [i for i, x in enumerate(inside_boundary) if x]
combined_uv_array = cleaned_boundary_uv_array.copy()
combined_uv_array.extend(inside_boundary_ids)

combined_uv_array_unique = []
[combined_uv_array_unique.append(x) for x in combined_uv_array
 if x not in combined_uv_array_unique]

combined_uv_array = np.array(combined_uv_array_unique)
location_uvs = np.array(uv_array[combined_uv_array])
triangulated_uvs = Delaunay(location_uvs)

# Eliminate Triangles that Only Connect to Border Points
highest_boundary_index = len(combined_uv_array) - len(inside_boundary_ids)
triangles = np.array(triangulated_uvs.simplices)
reduced_triangles = []
for row in triangles:
    if np.all(row > highest_boundary_index):
        reduced_triangles.append(row)

reduced_triangles = np.array(reduced_triangles)

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
    face_data_reduced[face_data_reduced['uv']
                      .isin(combined_uv_array)]["vertex"].to_numpy()

vertex_ids = np.empty((len(combined_uv_array))).astype(int)

for index, value in enumerate(indicies_of_sorted_indicies):
    vertex_ids[value] = sorted_vertex_ids[index]

# Get the Vertex Positions using the Indicies
location_surface = np.array(mesh_verticies[vertex_ids])

# Find the Surface Area of the 3D Location Drawing
# Find the lengths of the triangle sides (used for outlier detection later)
sides_array = np.empty((0, 3))
for tri in reduced_triangles:
    p1 = location_surface[tri[0]]
    p2 = location_surface[tri[1]]
    p3 = location_surface[tri[2]]
    sides_array = np.concatenate((sides_array,
                                  find_triangle_sides(p1, p2, p3)), axis=0)
# Calculate the Area Using Heron's Formula
area_array = np.array([])
for sides in sides_array:
    area_array = np.append(area_array, heron_area(sides))

# Remore large triangles as outliers
deviations = 50
cleaned_area_array = remove_area_outliers(area_array, deviations)
cleaned_sides_array = remove_side_outliers(sides_array, deviations)
print(f"The area of the location is {np.sum(cleaned_area_array)} scene units" +
      f" after removing area outliers {deviations} deviations away")
sides_area_array = np.array([])
for sides in cleaned_sides_array:
    sides_area_array = np.append(sides_area_array, heron_area(sides))
print(f"The area of the location is {np.sum(sides_area_array)} scene units" +
      f" after removing side length outliers {deviations} deviations away")
