'''
A library for analyzing 2D location drawings with a 3D mesh.

This library allows for the import of mesh data created in an 3D modeling/CAD
program and saved as the common OBJ format. This mesh is mapped to a drawing
template for where subjects feel touch perception and this library allows the
user to find the path and path length between any set of points as well as the
surface area of a drawn location on the 3D model.

Modules
-------
data_manager :
    Converts input mesh data to python usable data tables.

Methods
-------
create_combined_data :
    Takes separate UV maps and lookup tables and combines them.
find_moved_uv_indicies :
    Return only the UVs that are within the location drawing bounds.
get_mesh_data :
    Saves out the mesh data from an obj file.
obj_to_txt :
    Takes in an OBJ mesh file and converts it into a text file.
txt_to_dataframe :
    Parses 3D mesh file in text format into the UV data and face lookup table.
'''
