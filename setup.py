from setuptools import setup

setup(
    name="Geodesic Path Finder",
    version="1.0.0",
    author="Nabeel Chowdhury",
    author_email="nabeel.chowdhury@case.edu",
    py_modules=["Geodesic Path Finder"],
    description="App for finding the geodesic for two points on a mesh",
    long_description='''The app included finds the distance between two points
    on a 3D mesh given their
    location on a 2D drawing. These 2D locations are translated to a 3D
    location by mapping the 3D mesh faces in x, y, and z to u and v
    coordinates on the 2D drawing. The app can output a visualization of the
    distances as well and show the path between the locations.''',
    license="MIT",
    keywords=["Geodesic", "Location Drawing", "Heat Method"],
    download_url="https://github.com/Nabizzle/Geodesic-Path-Finder",
    install_requires=["customtkinter==5.1.2", "opencv-python==4.7.0.72",
                      "numpy==1.24.2", "pandas==1.5.3", "polyscope==1.3.1",
                      "potpourri3d==0.0.8", "pynput==1.7.6",
                      "jupyterlab==3.5.3", "scipy==1.10.1",
                      "matplotlib==3.7.2", "ipympl==0.9.3", "pyvista==0.41.1",
                      "trame==3.1.0", "trame-vtk==2.5.8",
                      "trame-vuetify==2.3.1", "hydra-core==1.3.2",
                      "polars==0.20.3", "pyarrow==14.0.2"],
    platforms="windows",
)
