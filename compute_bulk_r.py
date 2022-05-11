#  The code for the manuscript: 
#  LAWS: Local Alignment for Water Sites - tracking ordered water in simulations
#  Authors: Eugene Klyshko, Justin Kim, and Sarah Rauscher
#  References in the code are related to the main text of the manuscript
# 
#  Developer: Eugene Klyshko

# This code generates bulk water sites (> 6 A from the protein) and computes offset vectors from the nearest water molecules to each bulk water site
# in each frame of the MD simulation

## (!) User should adjust these parameters according to their system and file names (1-2):

# 1. Path to a simulation structure and trajectory (which includes positions of all water molecules)
traj_folder = './'
struct = traj_folder + 'firstframe.gro'
trajectory = traj_folder + 'trajectory.xtc'

# 2. Parameters of the system and trajectory
stride = 10 # Stride for analysis (when stride=10 we will analyze only every 10-th frame of the original trajectory)
N_chains = 4 # Number of symmetric chains in the simulation. In the manuscript, we have a unit cell with 4 protein chains. 
N_atoms_in_chain = 1473 # Number of protein atoms in each chain.
n_waters = 120 # Number of bulk water sites to generate

# import useful libraries, such as MDAnalysis, and necessary functions
import MDAnalysis as md
import numpy as np
import sys
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC, augment_coordinates

# import function necessary for analysis of a simulation
from laws import (
    find_chains,
    find_offsets
)

# This function assumes that atomic coordinates are written consequtively (chain by chain) in the structure file from the very beginning of the file.
# It creates MDAnalysis selection for each chain (can be applied to both CRYSTAL_STRUCTURE and MD struct) for further analysis
if N_chains >= 1:
    chains = find_chains(N_chains, N_atoms_in_chain)

# Loading the system into MDAnalysis universe:
traj = md.Universe(struct, trajectory)
print("Information about trajectory", trajectory)
frames = len(traj.trajectory) # Determine number of frames in the trajectory
timesteps = len(range(0, frames, stride)) - 1 # Determine number of frames to be used in analysis, including only every stride-th frame

# memory allocation for offset vectors, distances (magnitudes of offset vectors)
distances = np.empty((timesteps, n_waters), dtype=np.float)
offsets = np.empty((timesteps, n_waters, 3), dtype=np.float)

## Pipeline: 
all_waters = traj.select_atoms('name OW') # selecting all water oxygens
protein_atoms = traj.select_atoms('protein') # selecting all protein atoms

# Bulk water sites are generated as the positions at least 6 A from the protein. 
# A nice quick trick to do it with MDAnalysis: (i) choose water molecules which are more than 6 A from the protein (in the first frame), 
# and (ii) use their positions (in the first frame) as bulk water sites. 
# Note (!) make sure there are more of such waters than your desired number in n_waters. Selecting the first n_waters and their coordinates:
bulk_water_sites  =  traj.select_atoms('name OW and not around 6.0 (protein and not type H)').atoms[:n_waters].positions.copy()

# Looping over all time frames
for t, ts in enumerate(traj.trajectory[0:frames:stride]):
    
    # Computing distances to the closest water taking into account preiodic images
    box = ts.dimensions # box dimensions for treating periodic boundary conditions
    dist_mtx = distance_array(
        bulk_water_sites,
        all_waters.atoms.positions,
        box=box,
        backend='OpenMP'
    )
    distances[t] = np.min(dist_mtx, axis=1)
    
    # Finding offset vectors to the closest water taking into account preiodic images
    nearest_water_indeces = np.argmin(dist_mtx, axis=1)
    nearest_water_positions = all_waters.atoms[nearest_water_indeces].positions
    offsets[t] = find_offsets(
        nearest_water_positions,
        bulk_water_sites.astype(np.float32),
        dist_mtx,
        box
    )
    
    # printing progress
    if t % 10 == 0:
        print("Timestep: {} out of {}".format(t, timesteps))

# filename to save all the results in npy array
filename_to_save = 'bulk'
np.save(filename_to_save + '_offsets', offsets) # Offset vectors r, shape (chains, frames, 3)
np.save(filename_to_save + '_distances', distances) # Magnitudes of offset vectors, shape (chains, frames, 1)
