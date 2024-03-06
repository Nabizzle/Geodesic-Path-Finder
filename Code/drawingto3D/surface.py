'''
Takes a location drawing and converts it into a 3D surface.

Used the mesh data made for each sex and side of the body to take a location
drawing from the drawing template and create a 3D surface connected to pyvista.

Methods
-------
clean_uv_border : Removes any duplicated UV points in an array.
convert_uv_to_vertex :
    Take the UVs in the location drawing and return the verticies surface.
create_surface : Create a 3D surface from the location drawing verticies.
find_enclosed_uvs : Find all UVs contained by the location drawing.
find_uv_indicies :
    Converts the location drawing border pixels to the nearest UV values.

Notes
-----
The surface created here is connected to
`pyvista <https://docs.pyvista.org/version/stable/>`_ and can be used with any
of the functions contained in that library.
'''
import polars as pl
import matplotlib.path as mpltPath
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree
from typing import List


def clean_uv_border(boundary_uv_array: np.ndarray) -> List[int]:
    '''
    Removes any duplicated UV points in an array.

    Parameters
    ----------
    boundary_uv_array : np.ndarray
        A 1D array that lists all the row number of the UV array table that are
        part of the border around a location drawing.

    Returns
    -------
    clean_uv_border : List[int]
        The UV border of a location drawing with no duplicate UV points.

    See Also
    --------
    find_uv_indicies :
        Converts the location drawing border pixels to the nearest UV values.
    '''
    cleaned_boundary_uv_array = []
    [cleaned_boundary_uv_array.append(x)
     for x in boundary_uv_array if x not in cleaned_boundary_uv_array]

    return cleaned_boundary_uv_array


def convert_uv_to_vertex(uv_indicies: np.ndarray, lookup_data: pl.DataFrame,
                         mesh_verticies: np.ndarray) -> np.ndarray:
    '''
    Take the UVs in the location drawing and return the verticies surface.

    Takes the row values of the UVs contained by the location drawing and
    find the corresponding verticies of the 3D mesh that make up the surface
    made by the location drawing in 3D.

    Parameters
    ----------
    uv_indicies : np.ndarray
        The row values of the UVs that make up the location drawing.
    lookup_data : pl.DataFrame
        The lookup table for finding which UVs go to which verticies.
    mesh_verticies : np.ndarray
        A Nx3 array of each vertex value.

    Returns
    -------
    location_surface : np.ndarray
        A Nx3 array of each vertex that make up the drawn location in 3D.
    '''
    vertex_ids = convert_uv_to_vertex_id(uv_indicies, lookup_data)

    # Get the values of all of the verticies
    location_surface = np.array(mesh_verticies[vertex_ids])

    return location_surface


def convert_uv_to_vertex_id(uv_indicies: np.ndarray, lookup_data: pl.DataFrame,
                            ) -> np.ndarray:
    '''
    Take the UVs in the location drawing and return the verticies surface.

    Takes the row values of the UVs contained by the location drawing and
    find the corresponding verticies of the 3D mesh that make up the surface
    made by the location drawing in 3D.

    Parameters
    ----------
    uv_indicies : np.ndarray
        The row values of the UVs that make up the location drawing.
    lookup_data : pl.DataFrame
        The lookup table for finding which UVs go to which verticies.

    Returns
    -------
    location_surface : np.ndarray
        A Nx3 array of each vertex that make up the drawn location in 3D.
    '''
    # Sort the UV indicies
    lookup_data_reduced = lookup_data[["vertex", "uv"]].unique()
    lookup_data_reduced = lookup_data_reduced.sort(by=['uv'])
    sorted_vertex_ids = lookup_data_reduced.filter(
        lookup_data_reduced['uv'].is_in(uv_indicies)
        )["vertex"].to_numpy()

    # Find all indicies for each vertex and organize it by the sort
    vertex_ids = np.empty((len(uv_indicies))).astype(int)
    indicies_of_sorted_indicies = np.argsort(uv_indicies)
    for index, value in enumerate(indicies_of_sorted_indicies):
        vertex_ids[value] = sorted_vertex_ids[index]

    return vertex_ids


def create_surface(location_surface: np.ndarray) -> pv.PolyData:
    '''
    Create a 3D surface from the location drawing verticies.

    Parameters
    ----------
    location_surface : np.ndarray
        A Nx3 array of each vertex that make up the drawn location in 3D.

    Returns
    -------
    shell : pv.PolyData
        The 3D mesh from the location drawing.
    '''
    # Convert the mesh into a point cloud
    cloud = pv.PolyData(location_surface)
    # Triangulate the 3D Mesh with short triangles
    volume = cloud.delaunay_2d(alpha=0.05)
    shell = volume.extract_geometry()

    return shell


def find_enclosed_uvs(uv_array: np.ndarray,
                      clean_boundary_uv_array: List[int]) -> np.ndarray:
    '''
    Find all UVs contained by the location drawing.

    Parameters
    ----------
    uv_array : np.ndarray
        Table of x, y positions of every uv point for the 3D mesh.
    clean_boundary_uv_array : List[int]
        A list of row numbers in the `uv_array` without duplicates.

    Returns
    -------
    combined_uv_indicies : np.ndarray
        Row numbers of UV values inside of and on the location drawing.

    See Also
    --------
    clean_uv_border : Removes any duplicated UV points in an array.
    find_uv_indicies :
        Converts the location drawing border pixels to the nearest UV values.

    Notes
    -----
    Uses the drawn location to bored an enclosed border around a region on the
    UV map. Then finds all the UVs inside of the border for use in extracting
    that portion of the mesh from the 3D model.
    '''
    # Make a boundary path from the boundary UVs
    path = uv_array[clean_boundary_uv_array]
    boundary = mpltPath.Path(path)

    # Find the UVs that lie inside the UV boundary
    inside_boundary = boundary.contains_points(uv_array)
    inside_boundary_ids = [i for i, x in enumerate(inside_boundary) if x]
    combined_uv_indicies = clean_boundary_uv_array.copy()

    # Add the contained UVs to the boundary UV array
    combined_uv_indicies.extend(inside_boundary_ids)

    # Eliminate any double counted UVs
    combined_uv_array_unique = []
    [combined_uv_array_unique.append(x)
     for x in combined_uv_indicies if x not in combined_uv_array_unique]

    # Find the UVs of the solid location drawing
    combined_uv_indicies = np.array(combined_uv_array_unique)

    return combined_uv_indicies


def find_uv_indicies(border_points: pl.DataFrame, uv_array: np.ndarray,
                     image_x_size: int, image_y_size: int) -> np.ndarray:
    '''
    Converts the location drawing border pixels to the nearest UV values.

    Takes the location drawing pixel locations and converts it to the index the
    closest UV values to the pixel using a KD Tree.

    Parameters
    ----------
    border_points : np.ndarray
        The x and y pixel values of a border point
    image_x_size : int
        The x dimension of the location drawing image in pixels
    image_y_size : int
        The y dimension of the location drawing image in pixels

    Returns
    -------
    border_uvs : np.ndarray
        The row numbers of the closest uv to the 2D border point list
    '''
    data_array = border_points.to_numpy().astype(np.float16)
    data_array[:, 0] = data_array[:, 0] / image_x_size
    data_array[:, 1] = 1 - (data_array[:, 1] / image_y_size)
    _, border_uvs = KDTree(uv_array).query(data_array)
    return border_uvs
