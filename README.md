# Geodesic-Path-Finder
<img align="left" src=https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Media/App%20Icons/sheen%20robotic%20hand.png width=200>

Uses UV mapping of the right and left upper arms of male and female presenting 3D models to find the distances between drawn sensory locations from 2D hand and arm
images on the 3D arm images. This is done using the [Heat Method](https://dl.acm.org/doi/abs/10.1145/2516971.2516977) to find the distance from any given point to
any other point on the mesh. Given a starting sensory location, a distance array to every other vertex is found by using the Heat Method along the triangulated arm
mesh. The end vertex number is then used as the index of the distance array to extract out the geodesic distance between the input points. A visualization of the path
between the points is also found using [edge flips](https://dl.acm.org/doi/abs/10.1145/3414685.3417839). The mesh can be visualized using
[Polyscope](https://polyscope.run/py/) in python.

To convert from the 2D image of location drawings to the 3D x, y, z position, the mesh is [UV mapped](#uv-mapping) to the location drawing so that x and y pxiel values
on the drawing correspoind to x, y, z vertex values on the 3D mesh.

> **Note**
>
> The edge flip method of showing a path does not guarentee the shortest path along the mesh, it is just a short path. The Heat Method distance should be used for any
> distance metric.

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
mesh is in the [Right Hand Path Test notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/Right%20Hand%20Path%20Test.ipynb) and a simple example
on a sphere is in the [Simple Geodesic Path Test notebook](https://github.com/Nabizzle/Geodesic-Path-Finder/blob/main/Code/Simple%20Geodesic%20Path%20Test.ipynb).

# UV Mapping

# Acknowledgments
<a href="https://www.flaticon.com/free-icons/technology" title="technology icons">App Icon from winnievinzence</a>
