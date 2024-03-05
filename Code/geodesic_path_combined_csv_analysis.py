'''
geodesic_path_combined_csv_analysis.py

The purpose of this script is to analyze geodesic distances between centroids
from data within multiple csv files that contain data from multiple
participants and contacts. For each study participant, the script will load
a unique mesh for his amputation side.

Based on code written by Nabeel Chowdhury
Last edit: Sedona Cady 9/27/2023

'''

# Load in libraries
import numpy as np
import polars as pl
from tkinter import filedialog
import os
from pathlib import Path
import sys
from drawingto3D.geodesic_path import GeodesicPath
import re

# Add geodesic path finder to path
sys.path.append("D:/GitHub/Geodesic-Path-Finder")

# Load in border(s) from csv file(s) (Sprompt user)
filenames =\
    filedialog.askopenfilenames(
        title="Please select .csv file of UV map centroid coordinates",
        filetypes=[("CSV Files", "*.csv")])

# Specify subject hand sides
SID = ["S102", "S104", "S106", "S107", "S109", "RCT01", "RCT02"]
amp_side = ["Right hand", "Right hand", "Left hand", "Left hand", "Left hand",
            "Right hand", "Left hand"]

# Calculate path distances for each file
for file in filenames:
    all_data = pl.read_csv(file)
    location_data =\
        pl.read_csv(file).drop_nulls()[
            ["start x", "start y",
             "end x", "end y"]].to_numpy()
    if (len(location_data) < 1):
        print('No data, skipping...')
        continue
    distance_vec = np.tile(0, (len(location_data), 1))
    sex = "male"
    sid_trial_match = re.search('(.+?)_', all_data["Contact"][0])
    sid_trial = sid_trial_match.group(1)
    amp_side_trial = amp_side[SID.index(sid_trial)]
    if amp_side_trial == "Right hand":
        side = "right"
    else:
        side = "left"
    print(side)
    GP = GeodesicPath(sex, side)
    GP.load_data(location_data)
    GP.found_distances = GP.calculate_distances()

    distance_vec = GP.found_distances
    all_data = all_data.with_columns(
        pl.lit(distance_vec).alias("Distance")
    )
    thisfile = os.path.basename(file)
    saveoutpath = "../Data/output/"
    saveoutfilename = 'ComputedDistances_' + thisfile
    fullfilepath = Path(f"{saveoutpath}/{saveoutfilename}")
    fullfilepath.parent.mkdir(exist_ok=True, parents=True)
    with open(fullfilepath, "w") as f:
        all_data.write_csv(f, include_header=True)
