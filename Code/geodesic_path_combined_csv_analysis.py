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
from tkinter import messagebox
import numpy as np
import potpourri3d as pp3d
import pandas as pd
import cv2
from typing import Dict
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import os
import ntpath
import sys
from geodesic_path import GeodesicPath
import re

# Add geodesic path finder to path
try:
    sys.path.append("D:\GitHub\Geodesic-Path-Finder")
except:
    folderpath = askopenfilename.askdirectory()
    sys.path.append(folderpath)

# Load in border(s) from csv file(s) (Sprompt user)
filenames = filedialog.askopenfilenames(title="Please select .csv file of UV map centroid coordinates", filetypes=[("CSV Files", "*.csv")])

# Specify subject hand sides
SID = ["S102", "S104", "S106", "S107", "S109", "RCT01", "RCT02"]
amp_side = ["Right hand", "Right hand", "Left hand", "Left hand", "Left hand", "Right hand", "Left hand"]

# Calculate path distances for each file
for f in range(0,len(filenames)):
    all_data = pd.read_csv(filenames[f])
    location_data =\
                pd.read_csv(filenames[f])[["start x", "start y",
                                    "end x", "end y"]].dropna().to_numpy()
    distance_vec = np.tile(0,(len(location_data),1))
    sex = "male"
    sid_trial_match = re.search('(.+?)_',all_data.Contact[0])
    sid_trial = sid_trial_match.group(1)
    amp_side_trial = amp_side[SID.index(sid_trial)]
    if amp_side_trial=="Right hand":
        side = "right"
    else:
        side = "left"
    print(side)
    GP = GeodesicPath(sex,side)
    GP.load_mesh()
    if (len(location_data) > 1):
        GP.analyze_data(location_data)
    else:
        GP.start_x_location = location_data[0,0]
        GP.start_y_location = location_data[0,1]
        GP.end_x_location = location_data[0,2]
        GP.end_y_location = location_data[0,3]
        GP.found_distances = GP.calculate_distances()
        GP.found_paths = GP.calculate_paths()
    distance_vec = GP.found_distances
    all_data['Distance'] = distance_vec
    thisfile = os.path.basename(filenames[f])
    saveoutpath = "../Data/output/"
    saveoutfilename = 'ComputedDistances_'+thisfile
    fullfilename = os.path.join(saveoutpath,saveoutfilename)
    fid = open(fullfilename,"w")
    all_data.to_csv(fid,sep=',',index=False,header=True)
    fid.close()

