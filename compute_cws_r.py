import warnings
warnings.filterwarnings('ignore')

import MDAnalysis as md
import numpy as np
import sys
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC, augment_coordinates
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.topology.guessers import guess_bonds

# algorithmic part
from laws import (
    solve3Dtrilateration_linear, 
    solve3Dtrilateration_lsq_linear,
    f_i, 
    JTJ_i,
    JTf_i,
    calc_S,
    solve3Dtrilateration_nonlinear,
    solve3Dtrilateration_nonlinear_python,
    solve3Dtrilateration_nonlinear_python_lm
)

# simulation part
from laws import (
    find_N_closest_heavy_atoms,
    find_chains,
    create_connectors,
    find_pbc_coords,
    apply_correction,
    find_offsets,
    visualize_step
)

from mpi4py import MPI

if len(sys.argv) <= 3:
    print("Needs 3 arguments!")
    exit(0)

folder = sys.argv[1]
file_to_save = sys.argv[2]
folder_to_save = sys.argv[3]

struct = folder + 'firstframe_fix.gro'
trajectory = folder + 'traj_fix.xtc'
traj = md.Universe(struct, trajectory)
save_top = md.Universe(struct)
print(trajectory)
sys.stdout.flush()

frames = len(traj.trajectory) # 100001
stride = 10
timesteps = len(range(0, frames, stride)) - 1 # =1000

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

chains = find_chains()
N_chains = 4
N_atoms_in_chain = 1473
n_waters = 94

local_size = timesteps // size # make sure timesteps equally divided between ranks (!)
ind_start = rank * local_size
traj_start = ind_start * stride

if rank != size - 1 :
    ind_end = (rank + 1) * local_size
    traj_end = ind_end * stride
else:
    ind_end = timesteps
    local_size = timesteps - ind_start
    traj_end = ind_end * stride

    
print(rank, local_size, ind_start, ind_end, timesteps)
sys.stdout.flush()

if rank == 0:
    D_crystal = np.empty((timesteps, N_chains, n_waters), dtype=np.float)
    E_error = np.empty((timesteps, N_chains,  n_waters), dtype=np.float)
    S_crystal = np.empty((timesteps, N_chains, n_waters, 3), dtype=np.float)
else:
    D_crystal = None
    E_error = None
    S_crystal = None

local_D_crystal = np.zeros((local_size, N_chains, n_waters), dtype=np.float)
local_S_crystal = np.zeros((local_size, N_chains, n_waters, 3), dtype=np.float)
local_E_error = np.zeros((local_size, N_chains, n_waters), dtype=np.float)

single_pdz = md.Universe('7_crystal_waters/4pdz_water.gro')
heavy_atoms = single_pdz.select_atoms('protein and not type H')

crystal_box = single_pdz.trajectory[0].dimensions

crystal_heavy_atoms_4 = [
    single_pdz.select_atoms('protein and not type H ' + chain) for chain in chains
]

crystal_waters_4 = [
    single_pdz.select_atoms('resname HOH and name OW and index {}:{}'.format(
        N_atoms_in_chain * N_chains + 3 * n_waters * chain,
        N_atoms_in_chain * N_chains + 3 * n_waters * (chain + 1) - 1)
    ) for chain in range(N_chains) 
]

connectors_set = create_connectors(crystal_waters_4, crystal_heavy_atoms_4, crystal_box)

crystal_waters_info = [
        [
            find_N_closest_heavy_atoms(water.position, heavy_atoms, crystal_box) for water in crystal_waters_4[chain]
        ]
    for chain in range(N_chains)
]
selectors = [
    [
        [
            'index {} and resid {} and name {}'.format(at[0], at[1], at[2]) for at in atoms
        ]
        for atoms, _, _ in crystal_waters_info[chain]
    ]
    for chain in range(N_chains)         
]
heavy_distances = [
    [d for _, _, d in crystal_waters_info[chain]]
    for chain in range(N_chains)
]

all_waters = traj.select_atoms('name OW') # and around 6.0 (protein {})'.format(chain_sel), updating=True)
all_chains_protein = traj.select_atoms('protein')

# memory allocation
crystal_water_positions = np.empty((n_waters, 3))
crystal_water_errors = np.empty((n_waters))

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
                crystal_water_errors[m] = er #/ len(heavy_distances[j][m])
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
            
            # Recording Errors
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

if rank == 0:
    print('Tunneled the barrier')
    sys.stdout.flush()
    
comm.Gather(local_D_crystal, D_crystal, root=0)
comm.Gather(local_E_error, E_error, root=0)
comm.Gather(local_S_crystal, S_crystal, root=0)

## save or print
if rank == 0:
    print(D_crystal.shape)
    np.save(file_to_save + '_distances', D_crystal)
    np.save(file_to_save + '_errors', E_error)
    np.save(file_to_save + '_shifts', S_crystal)
    
    print('Saved to ' + file_to_save)