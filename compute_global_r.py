#  The code for the manuscript: 
#  LAWS: Local Alignment for Water Sites - tracking ordered water in simulations
#  Authors: Eugene Klyshko, Justin Kim, and Sarah Rauscher
#  References in the code are related to the main text of the manuscript
# 
#  Developer: Eugene Klyshko

# This code utilizes global alignment method to compute offset vectors from the nearest water molecules to each CWS
# in each frame of the MD simulation

## (!) User should adjust these parameters according to their system and file names (1-3):

# 1. Path to a simulation structure and trajectory (which includes positions of all water molecules)
traj_folder = './'
struct = traj_folder + 'firstframe.gro'
trajectory = traj_folder + 'trajectory.xtc'

# 2. Path to the structure of the system (single protein, unit cell or supercell) containing only protein and CWS. 
# Typically, a crystal structure (PDB or GRO file) from which your simulation system was built.
# In case of the MD simulation of a protein crystal (as in the manuscript), the unit cell structure was constructed using option `build a unit cell` in CHARMM-GUI server,
# by also preserving all crystallographic water oxygens. 
# Note: Protein atoms numbering must be the same as in your trajectory file. It is normally the case since the MD system is constructed consequtively: 
# PDB (protein + CWS coordinates) -> solvating the system with H2O (using for example, gmx solvate) -> adding ions.
# Initial information for multilateration (coordinating protein atoms for each CWS) will be extracted from this structure file. 
CRYSTAL_STRUCTURE = 'crystal.gro' # Protein atom numbers in this file should correpond to 'firstframe.gro'

# 3. Parameters of the system and trajectory
stride = 10 # Stride for analysis (when stride=10 we will analyze only every 10-th frame of the original trajectory)
N_chains = 4 # Number of symmetric chains in the simulation. In the manuscript, we have a unit cell with 4 protein chains. 
N_atoms_in_chain = 1473 # Number of protein atoms in each chain.
n_waters = 94 # Number of CWS in the crystal structure.


# import useful libraries, such as MDAnalysis, and necessary functions
import numpy as np
import sys, os
import MDAnalysis as md
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array

# import function necessary for analysis of a simulation
from laws import (
    find_chains,
    find_offsets
)

# Using a parallel (MPI) to speed up the computations across the protein chains
from mpi4py import MPI


# This function assumes that atomic coordinates are written consequtively (chain by chain) in the structure file from the very beginning of the file.
# It creates MDAnalysis selection for each chain (can be applied to both CRYSTAL_STRUCTURE and MD struct) for further analysis
if N_chains >= 1:
    chains = find_chains(N_chains, N_atoms_in_chain)


# Loading the system into MDAnalysis universe:
traj = md.Universe(struct, trajectory)
print("Information about trajectory", trajectory)
frames = len(traj.trajectory) # Determine number of frames in the trajectory
timesteps = len(range(0, frames, stride)) - 1 # Determine number of frames to be used in analysis, including only every stride-th frame

# Loading the crystal structure into MDAnalysis universe:
crystal = md.Universe(CRYSTAL_STRUCTURE)
crystal_box = crystal.trajectory[0].dimensions # dimensions of the sell. Needed for treating periodic boundaries.

# MPI initialization:
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# MPI memory allocation to each process:
# We divide computations between ranks across the chains, so each process will analyze its own chain's trajectory
# Setting up local indexing and sizes

local_n = N_chains // size
ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

# Next block of code describes memory allocation to each process (for MPI):
if rank == 0:
    # main global variables containing offset vectors, distances (magnitudes of offset vectors)
    distances = np.empty((N, timesteps, n_waters), dtype=np.float)
    offsets = np.empty((N, timesteps, n_waters, 3), dtype=np.float)
else:
    distances = None
    offsets = None

# same variables but local for each process
local_distances = np.empty((local_n, timesteps, n_waters), dtype=np.float)
local_offsets = np.empty((local_n, timesteps, n_waters, 3), dtype=np.float)


# Pipeline: 

crystal_waters = crystal.select_atoms('name OW') # selecting crystal waters

# Main loop over chains (each process analyses bulk water sites arou)
for i, j in enumerate(range(ind_start, ind_end)):
    
    chain_sel = chains[j] 
    chain = traj.select_atoms('protein and not type H' + chain_sel) # selecting a specific chain
    sel_text = 'name OW and around 6.0 protein {}'.format(chain_sel)
    chain_waters = traj.select_atoms(sel_text, updating=True) # select water molecules in a simulation within 6 A from the protein
    
    protein_atoms = crystal.select_atoms('protein and not type H') # select protein atoms in the crystal
    
    # Looping over all time steps
    for t, ts in enumerate(traj.trajectory[0:frames:stride]):
        
        # printint progress
        if t % 10 == 0: 
            print("Rank: {}, timestep: {} out of {}".format(rank, t, timesteps))
            sys.stdout.flush()
            
        align.alignto(protein_atoms, chain, weights="mass") # globally aligning protein structure onto the crystal structure
        
        # now positions of crystallographic oxygens are new CWS positions
        box = ts.dimensions
        dist_mtx = np.empty((n_waters, chain_waters.n_atoms), dtype=np.float) # finding the nearest neighbour water oxygen for each CWS
        distance_array(
            crystal_waters.atoms.positions,
            chain_waters.atoms.positions,
            box=box,
            result=dist_mtx,
            backend='OpenMP'
        )
        local_distances[i, t, :] = np.min(dist_mtx, axis=1)
 
        # Finding offset vectors to the closest water 
        water_indeces = np.argmin(dist_mtx, axis=1)
        positions_water = chain_waters.atoms[water_indeces].positions
        positions_crystal = crystal_waters.atoms.positions
        
        local_offsets[i, t, :] = find_offsets(
            positions_water,
            positions_crystal,
            dist_mtx,
            box
        )

# Collect data from all processes into rank 0 process to save into file
comm.Barrier()
comm.Gather(local_distances, distances, root=0)
comm.Gather(local_offsets, offsets, root=0)

## Saving files
if rank == 0:
    # filename to save all the results in npy array
    filename_to_save = 'global'
    np.save(filename_to_save + '_offsets', offsets) # Offset vectors r, shape (chains, frames, 3)
    np.save(filename_to_save + '_distances', distances) # Magnitudes of offset vectors, shape (chains, frames, 1)

