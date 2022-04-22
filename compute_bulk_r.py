#  The code for the manuscript: 
#  LAWS: Local Alignment for Water Sites - a method to analyze crystallographic water in simulations
#  Authors: Eugene Klyshko, Justin Kim, and Sarah Rauscher
#  Developer: Eugene Klyshko 
#  References in the code are related to the main text of the manuscript


## This code computes offset vectors for the bulk water sites

import numpy as np
import sys

import MDAnalysis as md
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array

from laws import (
    find_chains,
    find_offsets
)

from mpi4py import MPI


if len(sys.argv) <= 3:
    print("Needs 2 arguments! (1) folder with the structure and a trajectory, (2) filename to save files")
    exit(0)

folder = sys.argv[1]
file_to_save = sys.argv[2]

## Simulation structure and trajectory (including water molecules)
struct = folder + 'firstframe.gro'
trajectory = folder + 'trajectory.xtc'
traj = md.Universe(struct, trajectory)

chains = find_chains()

frames = len(traj.trajectory) 
start = 0
stride = 1
timesteps = len(range(0, frames, stride))
print(timesteps)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N = 4

# MPI memory allocation to each process:

## Setting up local indexing and sizes
local_n = N // size
ind_start = rank * local_n
if rank != size - 1 :
    ind_end = (rank+1)*local_n
else:
    ind_end = N

n_waters = 94

if rank == 0:
    D_crystal = np.empty((N, timesteps, n_waters), dtype=np.float)
    S_crystal = np.empty((N, timesteps, n_waters, 3), dtype=np.float)
else:
    D_crystal = None
    S_crystal = None
    
local_D_crystal = np.empty((local_n, timesteps, n_waters), dtype=np.float)
local_S_crystal = np.empty((local_n, timesteps, n_waters, 3), dtype=np.float)


# The structure of the crystal (unit cell or supercell) with all symmetric copies of CWS. Protein atoms numbering must be the same as in your structure/trajectory files

CRYSTAL_STRUCTURE = 'crystal.gro'
crystal = md.Universe(CRYSTAL_STRUCTURE)    
crystal_waters = crystal.select_atoms('name OW')

for i, j in enumerate(range(ind_start, ind_end)):
    chain_sel = chains[j]
    chain = traj.select_atoms('protein and not type H' + chain_sel)
    sel_text = 'name OW and around 6.0 protein {}'.format(chain_sel)
    chain_waters = traj.select_atoms(sel_text, updating=True)
    
    pdz_protein = crystal.select_atoms('protein and not type H')
    
    for t, ts in enumerate(traj.trajectory[start:frames:stride]):
        
        if t % 10 == 0:
            
            print("Rank: {}, timestep: {} out of {}".format(rank, t, timesteps))
            sys.stdout.flush()
            
        align.alignto(pdz_protein, chain, weights="mass")
        
        box = ts.dimensions
        
        dist_mtx = np.empty((n_waters, chain_waters.n_atoms), dtype=np.float)
        
        distance_array(
            crystal_waters.atoms.positions,
            chain_waters.atoms.positions,
            box=box,
            result=dist_mtx,
            backend='OpenMP'
        )
        local_D_crystal[i, t, :] = np.min(dist_mtx, axis=1)
        
        #assert np.all(np.min(dist_mtx, axis=1)) < 5.0, f"Distance to the closest exceeds 4.5 A: {np.min(dist_mtx, axis=1)}"
 
        # Finding offset vectors to the closest water 
        water_indeces = np.argmin(dist_mtx, axis=1)
        positions_water = chain_waters.atoms[water_indeces].positions
        positions_crystal = crystal_waters.atoms.positions
        
        local_S_crystal[i, t, :] = find_offsets(
            positions_water,
            positions_crystal,
            dist_mtx,
            box
        )
        
comm.Gather(local_D_crystal, D_crystal, root=0)
comm.Gather(local_S_crystal, S_crystal, root=0)

## save or print
if rank == 0:
    print(D_crystal.shape, S_crystal.shape)
    
    # Magnitudes of offset vectors, shape (chains, frames, 1)
    np.save(file_to_save + '_distances', D_crystal)
    # Offset vectors r, shape (chains, frames, 3)
    np.save(file_to_save + '_shifts', S_crystal)
    print('Crystal waters are saved to ' + file_to_save)
