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
geodesic_path.py :
    Finds the geodesic path between sets of points on a mesh
'''
