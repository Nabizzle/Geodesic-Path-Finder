
# Load in Libraries
from scipy.spatial import KDTree
import numpy as np
import matplotlib.path as mpltPath
import polars as pl
import cv2
import pyvista as pv
from hydra import initialize, compose


# Define Function to Find UV vertex indicies
def find_uv_index_kdtree(border_points: pl.DataFrame, image_x_size: int,
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
    data_array = border_points.to_numpy().astype(np.float16)
    data_array[:, 0] = data_array[:, 0] / image_x_size
    data_array[:, 1] = 1 - (data_array[:, 1] / image_y_size)
    _, indicies = KDTree(uv_array).query(data_array)
    return indicies


with initialize(version_base=None, config_path="config"):
    cfg = compose(config_name='config.yaml')
# Load in data
data = (
    pl.read_csv(f"{cfg.data_path}ExampleBoundaryTable.csv")
    .with_columns(pl.col("*").cast(pl.Int16))
    )

# Load in Male Right Arm Mesh Data
imported_data =\
    np.load(cfg.models.data)
mesh_verticies = imported_data["mesh_verticies"]
mesh_faces = imported_data["mesh_faces"]

# Load in uv data
uv_array = imported_data["uv_array"]

# import the face data
lookup_data = pl.DataFrame(imported_data["lookup_data"],
                           schema={"vertex": pl.UInt32, "uv": pl.UInt32})

# Load in the Male Right Arm and Find the image dimensions
img = cv2.imread(cfg.drawing_templates.image, 1)
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

# Translate the UV Triangulation to the 3D Mesh
# Sort the UV Indicies
indicies_of_sorted_indicies = np.argsort(combined_uv_array)
sorted_indicies = combined_uv_array[indicies_of_sorted_indicies]

# Make a look up table of one for one UV to Vertex Points
lookup_data_reduced = lookup_data[["vertex", "uv"]].unique()
lookup_data_reduced = lookup_data_reduced.sort(by=['uv'])

# Find all Verticies at Once and Reorganize the Array Based on the Sort
sorted_vertex_ids =\
    lookup_data_reduced.filter(lookup_data_reduced['uv']
                               .is_in(combined_uv_array))["vertex"].to_numpy()

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

# Print the Surface Area
print(f"The area of the drawn location is {shell.area} scene units")
