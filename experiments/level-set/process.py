
#%%
import os
import sys
import math
sys.setrecursionlimit(10000) 
import torch
import json
import pandas as pd
import numpy as np
import argparse
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.autograd.set_detect_anomaly(False)
# debug._set_state(False)

# script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
# src_dir = "/".join(script_dir.split("/")[:-2]) # src directory is two levels up
# sys.path.append(src_dir)

# if in jupyter notebook, use this instead
run_dir = os.getcwd()
src_dir = "/".join(run_dir.split("/")[:-2]) # src directory is two levels up
sys.path.append(src_dir)

#%%

# read csv
mat = np.loadtxt(f"{run_dir}/data/volcano_maungawhau.csv", delimiter=",") # 87 x 61

# generate meshgrid the same size as the matrix
x1 = np.linspace(0, 1, mat.shape[1]) # (61, )
x2 = np.linspace(0, 1, mat.shape[0]) # (87, )
mat = mat.flatten()
X1, X2 = np.meshgrid(x1, x2)
x_set = np.hstack([X1.reshape(-1, 1), X2.reshape(-1, 1)]) # (87*61, 2)
# plot the meshgrid

fig, ax = plt.subplots()
ax.contourf(X1, X2, mat.reshape(X1.shape), cmap='viridis')
# get colorbar
cbar = plt.colorbar(ax.contourf(X1, X2, mat.reshape(X1.shape), cmap='viridis'))
cbar.set_label('Elevation')
# get the proportion above 165
idx = np.where(mat > 165)[0]
# zero out the proportion above 165
mat[idx] = 0
# plot the new contour
fig, ax = plt.subplots()
ax.contourf(X1, X2, mat.reshape(X1.shape), cmap='viridis')
# get colorbar
cbar = plt.colorbar(ax.contourf(X1, X2, mat.reshape(X1.shape), cmap='viridis'))
cbar.set_label('Elevation')
plt.show()
#



x_to_elevation = {tuple(x): mat[i] for i, x in enumerate(x_set)}


#%%