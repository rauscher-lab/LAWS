
import MDAnalysis as md
import numpy as np
import sys
from MDAnalysis.analysis import align
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC, augment_coordinates

from laws import (
    find_chains,
    find_offsets
)

folder = sys.argv[1]
file_to_save = sys.argv[2]

struct = folder + 'firstframe.gro'
trajectory = folder + 'traj_whole_nojump_center_compact.xtc' 



print(trajectory)
sys.stdout.flush()
traj = md.Universe(struct, trajectory)

chains = find_chains(N_chains=108)

frames = len(traj.trajectory) # 100001
start = 0
stride = 10
timesteps = len(range(0, frames, stride)) # =1000

print(timesteps)

bulk_water_pos =  traj.select_atoms('name OW and not around 6.0 (protein and not type H)').atoms.positions.copy()
n_waters = bulk_water_pos.shape[0]
print(n_waters)

D_bulk = np.empty((timesteps, n_waters), dtype=np.float)
S_bulk = np.empty((timesteps, n_waters, 3), dtype=np.float)


chain_waters = traj.select_atoms('name OW', updating=False)
chain_protein = traj.select_atoms('protein')

bulk_water_positions = bulk_water_pos
    
    
for t, ts in enumerate(traj.trajectory[start:frames:stride]):
    box = ts.dimensions
    
    dist_mtx = distance_array(
        bulk_water_positions,
        chain_waters.atoms.positions,
        box=box,
        backend='OpenMP'
    )
    
    #assert np.any(np.min(dist_mtx, axis=1)) < 4.5, f"Distance to the closest exceeds 4.5 A: {np.min(dist_mtx, axis=1)}"
    
    D_bulk[t] = np.min(dist_mtx, axis=1)
    
    water_indeces = np.argmin(dist_mtx, axis=1)
    positions_water = chain_waters.atoms[water_indeces].positions
    positions_bulk = bulk_water_positions.astype(np.float32)

    S_bulk[t] = find_offsets(
        positions_water,
        positions_bulk,
        dist_mtx,
        box
    )

    if t % 10 == 0:
        print("Timestep: {} out of {}".format(t, timesteps))

np.save(file_to_save + 'bulk_distances', D_bulk)
np.save(file_to_save + 'bulk_shifts', S_bulk)
print('Crystal waters are saved to ' + file_to_save)
