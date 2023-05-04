# Geodesic-Path-Finder
<img align="left" src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/App%20Icons/sheen%20robotic%20hand.png width=200>

Uses UV mapping of the right and left upper arms of male and female presenting 3D models to find the distances between drawn sensory locations from 2D hand and arm
images on the 3D arm images. This is done using the [Heat Method](https://dl.acm.org/doi/abs/10.1145/2516971.2516977) to find the distance from any given point to
any other point on the mesh. Given a starting sensory location, a distance array to every other vertex is found by using the Heat Method along the triangulated arm
mesh. The end vertex number is then used as the index of the distance array to extract out the geodesic distance between the input points.
A visualization of the path between the points is also found using [edge flips](https://dl.acm.org/doi/abs/10.1145/3414685.3417839). The mesh can be visualized
using [Polyscope](https://polyscope.run/py/) in python.

To convert from the 2D image of location drawings to the 3D x, y, z position, the mesh is [UV mapped](#uv-mapping) to the location drawing so that x and y pxiel
values on the drawing correspoind to x, y, z vertex values on the 3D mesh.

> **Note**
>
> The edge flip method of showing a path does not guarentee the shortest path along the mesh, it is just a short path.
> The Heat Method distance should be used for any distance metric.

[![GitHub followers](https://img.shields.io/github/followers/Nabizzle?style=social)](https://github.com/Nabizzle)
![PyPI - Python Version](https://img.shields.io/badge/python-v3.9-blue)

# App Description
The app allow the user to find geodesic distance and path information in multiple ways.
![Geodesic Path Finder App](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/App%20Photo.png)

The top row allows the user to select between male and female right and left arms. These correspond to the right and left location drawing images. The load mesh
button must be pressed to load in the mesh data. For speed to loading, the mesh data has been presaved as compressed .npz files in the Data folder. Refer to the
[Save Mesh Data notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/Save%20Mesh%20Data.ipynb) for how the mesh data is extracted and saved.
> **Note**
> 
> The original location drawings appear to me modeled after a male body so the fit to the female form may not be perfect. Refer to the [UV Mapping](#uv-mapping)
> section for how the meshes were mapped to the location drawings.

The next two rows of the app allow the user to input in pixel values to two locations to measure between. The pixel values start from the top left corner and must
match up with the dimensions of the [right](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/right%20arm.png) and
[left](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/left%20arm.png) location drawings in the Media folder.

The fourth row of the app offers two additional ways of inputting starting and ending location information. The first two buttons in the row allow the user to click
on the location drawings to manually set start and end locations. An example of the right location image is below:

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/right%20arm.png width=300>

The final button of this row allow the user to load a table of starting and ending pixel values from a csv. The csv must start with headers for the start and end x
and y values. Any row with a missing value in the table is ommitted. An example table of values is below.

| start x | start y | end x | end y |
| ------- | ------- | ----- | ----- |
|   126   |   75    |  83   |  116  |
|   236   |   508   |  103  |  614  |
|   278   |   231   |  197  |  800  |
|   227   |   76    |  71   |  30   |
|   276   |   91    |  84   |  896  |

The final row of the app allow the user to caluculate the geodesic distances from the starting point(s) to the end point(s) and the paths between the points. These
calculations are output as .mat files to the output folder in the Data folder and added to the mesh visualization brought up by the third button in the row.
This button can be clicked at any time to see what has been added to polyscope. An example of a finished visualization with both the distance and path between
verticies is shown here.

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/Path%20of%20the%20Right%20Arm%20Mesh.png width=500>

A breakdown of how the app works behind the scenes is also in the
[UV to 3D Path notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/UV%20to%203D%20Path.ipynb). A more simple example on the male right arm
mesh is in the [Right Hand Path Test notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/Right%20Hand%20Path%20Test.ipynb) and a simple
example on a sphere is in the
[Simple Geodesic Path Test notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/Simple%20Geodesic%20Path%20Test.ipynb).

# UV Mapping
UV mapping of the meshes was done in [blender](https://www.blender.org/). This was first done by sculpting male and female anatomy from reference of which
[Anatomy for Sculptors](https://anatomy4sculptors.com/) was a major source. Once the musculature was sculted, the mesh was fit to the location drawings in two ways
depending on the sex of the model.

## Mapping the male mesh
For the male mesh, the model was scaled to fit with the location drawings as shown below:

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/Reference%20matching%20example.png width=500>

Once the proportions were correct, then seams in the mesh were created to match with landmarks on the location drawings. An example of this on the male hand is
below:

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/Example%20of%20making%20seams.png width=500>

Finally, these segments of the mesh are projected into the 2D space and moved into place over the location drawings as shown below for the hand:

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/UV%20Mapping%20example.png width=500>

## Mapping the Female Mesh
For the female mesh, the proportions of the body could not fit with the location drawings as the drawings are of a male figure. As a result, the female mesh had to
have this step skipped. The seams and mapping of the mesh to the drawings were made in the same way as above, but some areas had to be stretched to map to the
female body. Shown below is what that mapping looked like when the female mesh was skinned with the location drawings.

<img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/Female%20Mesh%20Blank.png width=400><img src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/Female%20Mesh%20Mapped.png width=400>

To avoid this issue mismatch between the drawings and the body, I would suggest female location drawings are made and the female mesh is mapped to them.

# Structure of an OBJ File
The meshes in this project were chosen to be .obj files because they have a nice human readable format that is explained in depth
[here](https://all3dp.com/1/obj-file-format-3d-printing-cad/). The main idea however is that the obj file is divded into sections for defining the mesh elements.

## Vertex Data
The first are all of the verticies were a line of the file has the format `v x y z` were `v` tell you that the line is for a vertex and the next three points are
the x, y, and z points in 3d space.

## Normal Data
The next relevant lines are designated at `vn x y z` were `vn` means those are the normal vectors of each face and the x, y, and z number are the x, y and z
magnitudes of the normal vector. These lines are not always necessary as the normal vectors can usually be recalcualted from the face data later in the file.

## UV Data
The lines that look like `vt u v` or `vt u v w` are the UV or texture data. These are the lines that contain were every vertex point is on the location map. These
maps are usually called texture maps as an image is wrapped on a mesh to give it more depth. The u coordinate is corresponds to the x direction on the image and the
v coordinate corresponds to the y direction on the image. sometimes there is a third column of data, w, which is a weighting information, but that is not relevant
for the meshes we use here.

## Face Data
Finally the rows with form `f v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3` are the culmination of all of the previous data to define the faces. The `f` defines these as face
rows and each point on a face references an index for a vertex, `v`, uv point, `vt`, and normal vector, `vn`, from the listed data defined above. A face can have 3 or more points that make it use and each column in these rows defines each point of the face.

> **Note**
> 
> For this code to work, all of the faces need to be triangles so our face data should only ever have three columns.

## Cleaning Up OBJ file for Use
In the code for saving out mesh data for preloading into the app or the rest of the code, the obj files are read as text files. In order to do this, they need to be
cleaned up slightly. OBJ files made in blender have comments starting with `#` that need to be deleted, calls to what autogenerated material is applied to the file,
and other grouping lines usually staring with `o` or `s`. These lines should be found at the top of the file and just before the lines for the faces.

# Requirements
- [Python 3.9.0](https://www.python.org/downloads/release/python-390/)
  - [potpourri3d](https://github.com/nmwsharp/potpourri3d) and [polyscope](https://polyscope.run/py/) do not work with newer versions of python
- [customtkinter version: 5.1.2](https://pypi.org/project/customtkinter/0.3/) ![dependency check for customtkinter](https://img.shields.io/librariesio/release/PyPi/customtkinter/5.1.2)
- [jupyterlab version: 3.5.3](https://pypi.org/project/jupyterlab/3.5.3) ![dependency check for jupyterlab](https://img.shields.io/librariesio/release/PyPi/jupyterlab/3.5.3)
- [numpy version: 1.24.2](https://pypi.org/project/numpy/1.24.2) ![dependency check for numpy](https://img.shields.io/librariesio/release/PyPi/numpy/1.24.2)
- [opencv-python version: 4.7.0.72](https://pypi.org/project/opencv-python/4.7.0.72) ![dependency check for opencv](https://img.shields.io/librariesio/release/PyPi/opencv-python/4.7.0.72)
- [pandas version: 1.5.3](https://pypi.org/project/pandas/1.5.3) ![dependency check for pandas](https://img.shields.io/librariesio/release/PyPi/pandas/1.5.3)
- [polyscope version: 1.3.1](https://pypi.org/project/polyscope/1.3.1) ![dependency check for polyscope](https://img.shields.io/librariesio/release/PyPi/polyscope/1.3.1)
- [potpourri3d version: 0.0.8](https://pypi.org/project/potpourri3d/0.0.8) ![dependency check for potpourri3d](https://img.shields.io/librariesio/release/PyPi/potpourri3d/0.0.8)
- [pynput version: 1.7.6](https://pypi.org/project/pynput/1.7.6) ![dependency check for pynput](https://img.shields.io/librariesio/release/PyPi/pynput/1.7.6)
- [scipy version: 1.10.1](https://pypi.org/project/scipy/1.10.1) ![dependency check for scipy](https://img.shields.io/librariesio/release/PyPi/scipy/1.10.1)

# Author
Code and documentation written by [Nabeel Chowdhury](https://www.nabeelchowdhury.com/)

# Acknowledgments
<a href="https://www.flaticon.com/free-icons/technology" title="technology icons">App Icon from winnievinzence</a>

[Anatomy for Sculptors](https://anatomy4sculptors.com/) for amazing references on anatomy.
