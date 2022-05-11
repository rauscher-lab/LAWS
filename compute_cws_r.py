#  The code for the manuscript: 
#  LAWS: Local Alignment for Water Sites - a method to analyze crystallographic water in simulations
#  Authors: Eugene Klyshko, Justin Kim, and Sarah Rauscher
#  References in the code are related to the main text of the manuscript
# 
#  Developer: Eugene Klyshko 
# This code computes offset vectors from the nearest water molecules to each crystallographic water sites
# in each frame of the MD simulation

## (!!!) User should adjust these parameters according to their system and file names:

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

# 4. Path to a folder where visualized CWS positions (as a new trajectory) will be saved. Can be left unchanged.
folder_to_save = 'visualize/'


# import useful libraries, such as MDAnalysis, and necessary functions
import numpy as np
import sys, os
import MDAnalysis as md
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC, augment_coordinates
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.topology.guessers import guess_bonds

# import trilateration part of the LAWS algorithm
from laws import ( 
    solve3Dtrilateration_nonlinear_python_lm
)

# import function necessary for analysis of a simulation
from laws import (
    find_N_closest_heavy_atoms,
    find_chains,
    create_connectors,
    find_pbc_coords,
    apply_correction,
    find_offsets,
    visualize_step
)

# Using a parallel (MPI) to speed up the computations across the timeframes
from mpi4py import MPI

# This function assumes that atomic coordinates are written consequtively (chain by chain) in the structure file from the very beginning of the file.
# It creates MDAnalysis selection for each chain (can be applied to both CRYSTAL_STRUCTURE and MD struct) for further analysis
if N_chains >= 1:
    chains = find_chains(N_chains, N_atoms_in_chain)

# Loading the system into MDAnalysis universe:
traj = md.Universe(struct, trajectory)
save_top = md.Universe(struct)
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

# We divide computations between ranks across the time, so each process will analyze its own part of the trajectory
local_size = timesteps // size # make sure timesteps equally divided between ranks (!!!)
ind_start = rank * local_size
traj_start = ind_start * stride

if rank != size - 1 :
    ind_end = (rank + 1) * local_size
    traj_end = ind_end * stride
else:
    ind_end = timesteps
    local_size = timesteps - ind_start
    traj_end = ind_end * stride

# Next block of code describes memory allocation to each process (for MPI):
print(rank, local_size, ind_start, ind_end, timesteps)
sys.stdout.flush() # printing stdout right away, no waiting for other processes to finish

if rank == 0:
    # main global variables containing offset vectors, distances (magnitudes of offset vectors) and LAWS 
    offsets = np.empty((timesteps, N_chains, n_waters, 3), dtype=np.float)
    distances = np.empty((timesteps, N_chains, n_waters), dtype=np.float)
    laws_errors = np.empty((timesteps, N_chains,  n_waters), dtype=np.float)
    
else:
    offsets = None
    distances = None
    laws_errors = None

# same variables but local for each process
local_distances = np.zeros((local_size, N_chains, n_waters), dtype=np.float)
local_offsets = np.zeros((local_size, N_chains, n_waters, 3), dtype=np.float)
local_laws_errors = np.zeros((local_size, N_chains, n_waters), dtype=np.float)


## Pipeline of the LAWS method

# 1. Selecting only heavy atoms in each chain
heavy_atoms = crystal.select_atoms('protein and not type H')
crystal_heavy_atoms_chains = [
    crystal.select_atoms('protein and not type H ' + chain) for chain in chains
]

## Detect CWS around each protein chain in the crystal structure
# Numbering of CWS must be consequtive (chain by chain) and should go after protein coordinates.
# Naming of CWS oxygens must be resname HOH, atom name OW.
crystal_waters_chains = [
    crystal.select_atoms('resname HOH and name OW and index {}:{}'.format(
        N_atoms_in_chain * N_chains + 3 * n_waters * chain,
        N_atoms_in_chain * N_chains + 3 * n_waters * (chain + 1) - 1)
    ) for chain in range(N_chains) 
]

# Find coordinating heavy atoms for each CWS around each chain 
crystal_waters_info = [
        [
            find_N_closest_heavy_atoms(water.position, heavy_atoms, crystal_box) for water in crystal_waters_chains[chain]
        ]
    for chain in range(N_chains)
]

# Find distances from each protein atom to CWS
heavy_distances = [
    [d for _, _, d in crystal_waters_info[chain]]
    for chain in range(N_chains)
]

# Create MDAnalysis selector for each heavy atom coordinating the CWS
selectors = [
    [
        [
            'index {} and resid {} and name {}'.format(at[0], at[1], at[2]) for at in atoms
        ]
        for atoms, _, _ in crystal_waters_info[chain]
    ]
    for chain in range(N_chains)         
]



all_waters = traj.select_atoms('name OW') 
all_chains_protein = traj.select_atoms('protein')

# memory allocation
crystal_water_positions = np.empty((n_waters, 3))
crystal_water_errors = np.empty((n_waters))

# Need or visualization
connectors_set = create_connectors(crystal_waters_chains, crystal_heavy_atoms_chains, crystal_box)

for j in range(N_chains):
    
    chain_sel = chains[j]
    
    heavy_atoms_sel = [all_chains_protein.select_atoms(*at_selector) for at_selector in selectors[j]]
    chain_protein = traj.select_atoms('protein ' + chain_sel)
    
    heavy_save_top_sel = [save_top.select_atoms('protein').select_atoms(*at_selector) for at_selector in selectors[j]]
    chain_protein_save_top = save_top.select_atoms('protein ' + chain_sel)
    chain_waters_save_top = save_top.select_atoms('name OW').atoms[:n_waters]
    protein_water_selection = chain_protein_save_top + chain_waters_save_top
    
    connectors = traj.atoms[connectors_set[j]]
    
    if rank == 0:
        protein_water_selection.write(f"{folder_to_save}water_sites_traj/protein_{j}.gro", multiframe=False)
        
    with md.Writer(
        f"{folder_to_save}temp/protein_{j}_{rank:03}.xtc",
        protein_water_selection.n_atoms,
        multiframe=True
    ) as W:
        
        for t, ts in enumerate(traj.trajectory[traj_start:traj_end:stride]):

            box = ts.dimensions
            save_top.trajectory[0].dimensions = box
            
            # LAWS Algorithm
            heavy_coords = apply_correction(heavy_atoms_sel, box)
            for m in range(n_waters):
                rr, er = solve3Dtrilateration_nonlinear_python_lm(heavy_coords[m], heavy_distances[j][m])
                crystal_water_positions[m] = rr
                crystal_water_errors[m] = er
                if er > 10:
                    print(m, j)
                    print(heavy_coords[m])
                    print(heavy_distances[j][m])
                    print(heavy_atoms_sel[m].atoms)
                    print(heavy_atoms_sel[m].positions)
            
            # Recording WS positions, i.e. visualization
            chain_protein_save_top.atoms.positions = chain_protein.atoms.positions.copy()
            visualize_step(crystal_water_positions, connectors, box, chain_waters_save_top)
            W.write(protein_water_selection)
            
            # Recording LAWS Errors
            local_E_error[t, j, :] = crystal_water_errors
            
            # Recording distances to closest waters 
            dist_mtx =  distance_array(
                crystal_water_positions,
                all_waters.atoms.positions,
                box=box,
            )
            assert np.all(np.min(dist_mtx, axis=1)) < 6.0, f"Distance to the closest exceeds 4.5 A: {np.min(dist_mtx, axis=1)}"
            local_D_crystal[t, j, :] = np.min(dist_mtx, axis=1)
            
            # Finding offset vectors to the closest water 
            water_indeces = np.argmin(dist_mtx, axis=1)
            positions_water = all_waters.atoms.positions[water_indeces]
            positions_crystal = crystal_water_positions.astype(np.float32)
            local_S_crystal[t, j, :] = find_offsets(
                positions_water,
                positions_crystal,
                dist_mtx,
                box
            )
            
            # Printing progress
            if t % 10 == 0:
                print("Rank: {}, chain: {}, timestep: {} out of {}".format(rank, j+1, t, local_size))
                sys.stdout.flush()

print("Rank: {} is ready".format(rank))
sys.stdout.flush()

comm.Barrier()
comm.Gather(local_offsets, offsets, root=0)
comm.Gather(local_distances, distances, root=0)
comm.Gather(local_laws_errors, laws_errors, root=0)

## Saving files
if rank == 0:
  
    # filename 
    filename_to_save = 'cws'
    np.save(filename_to_save + '_offsets', offsets) # Offset vectors r, shape (chains, frames, 3)
    np.save(filename_to_save + '_distances', distances) # Magnitudes of offset vectors, shape (chains, frames, 1)
    np.save(filename_to_save + '_laws_errors', laws_errors) # LAWS errors with shape (chains, frames, 1)
    
    print('Saved to ' + filename_to_save)
