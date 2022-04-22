#  The code for the manuscript: 
#  LAWS: Local Alignment for Water Sites - a method to analyze crystallographic water in simulations
#  Authors: Eugene Klyshko, Justin Kim, and Sarah Rauscher
#  Developer: Eugene Klyshko 
#  References in the code are related to the main text of the manuscript

import warnings
warnings.filterwarnings('ignore')

import numpy as np
from itertools import combinations

import MDAnalysis as md
from MDAnalysis.analysis.distances import distance_array, self_distance_array
from MDAnalysis.lib.mdamath import triclinic_vectors
from MDAnalysis.lib.distances import apply_PBC, augment_coordinates
from MDAnalysis.topology.guessers import guess_bonds
from MDAnalysis.analysis import align

from scipy.spatial.distance import squareform
from scipy.optimize import least_squares

import sys


def solve3Dtrilateration_linear(r_array, d_array):
    """Multilateration: Linear exact solver
    
    This function applies a linear excat solver for N=4 atoms. Most likely will not work.
    
    args: r_array - current positions of n protein atoms (n, 3)
          d_array - n distances from protein atoms as observed in a crystal structure (n)
          
    output: (x,y,z) - a new position of CWS
    """
    N_fixed = 4
    r_array = r_array[:N_fixed]
    d_array = d_array[:N_fixed]
    A = r_array[1:] - r_array[0]
    r1_2 = d_array[0]**2
    rn_2 = d_array[1:]**2
    dn_2 = np.linalg.norm(r_array[1:] - r_array[0], axis=1) **2
    bn = 0.5 * (r1_2 - rn_2 + dn_2)
    A_1 = np.linalg.inv(A)
    return np.dot(A_1, bn) + r_array[0]

def solve3Dtrilateration_lsq_linear(r_array, d_array):
    """Multilateration: Linear least squares estimator
    
    This function applies a linear least squares estimator. This function gets an initial approximation
    of the solution, which will be later refined with a nonlinear solver.
    
    args: r_array - current positions of n protein atoms, shape (n, 3)
          d_array - n distances from protein atoms as observed in a crystal structure, shape  (n)
          
    output: (x,y,z) - a new position of CWS
    """
    A = r_array[1:] - r_array[0]
    r1_2 = d_array[0]**2
    rn_2 = d_array[1:]**2
    dn_2 = np.linalg.norm(r_array[1:] - r_array[0], axis=1) **2
    bn = 0.5 * (r1_2 - rn_2 + dn_2)
    AtA = np.dot(np.transpose(A), A)
    k = np.linalg.cond(AtA)
    # if bad conditioned
    if k > 10**2:
        Q, R = np.linalg.qr(A)
        return np.dot(np.linalg.inv(R), np.dot(Q.T, bn)) + r_array[0]
    else:
        try:
            A_1 = np.linalg.inv(AtA)
            return np.dot(A_1, np.dot(A.T, bn)) + r_array[0]
        except:
            print("Can't invert in linear least sq. Returning mean of all coords")
            return np.mean(r_array, axis=0)
    

def f_i(r, ri, di):
  """Function to calculate a single residual in the LAWS error (Eq.2)"""
    return np.linalg.norm(r - ri) - di


def fun(r, r_array, d_array, weights=False):
  """Function to compute all weighted residuals in the LAWS error (Eq.2)"""
    if weights:
        D = np.sum(1. / d_array**2)
        w_array = (1. / d_array) / D**0.5 
    else: 
        D = len(d_array)
        w_array = np.ones(len(d_array)) / D**0.5
        
    residuals = []
    for rj, dj, wj in zip(r_array, d_array, w_array):
        residuals.append(wj * f_i(r, rj, dj))
    return np.array(residuals)


def solve3Dtrilateration_nonlinear_python_lm(r_array, d_array):
    """Multilateration: Non-linear solver 
    
    This function solves an optimization problem of (Eq.~2) by approximatin a solution with a linear optimizer and then
    refining the solution by Levenberg-Marquardt algorithm implemented in scipy.optimize `least_squres` function
    
    args: r_array - current positions of n protein atoms, shape (n, 3)
          d_array - n distances from protein atoms as observed in a crystal structure, shape  (n)
          
    output: solution.x - an optimum position of CWS, shape (3)
            squared sum of residuals - the value of the LAWS error (float) 
    """
    r0 = solve3Dtrilateration_lsq_linear(r_array, d_array)
    solution = least_squares(fun, r0, args=(r_array, d_array, True), gtol=1e-4, xtol=1e-4, ftol=1e-4)
    return solution.x, np.sum(solution.fun**2)


def find_N_closest_heavy_atoms(water_position, heavy_atoms, box, N_max=10, N_min=4, cutoff=4.51):
    """For a given position, the function finds a set of the closest atoms from the selection within a cutoff. 
    The initial run of the LAWS pipeline.
    
    args: water_position - current positions of n protein atoms, shape (n, 3)
          heavy_atoms - the MDAnalysis selection of all protein atoms in the universe
          box - MDAnalysis box, dimensions of the box to account for periodic boundaries, shape (6)
          N_max and N_min are the max and min number of protein atoms to be considered which coordinate a CWS, int
          cutoff - a cutoff distance in angstroms to be search for protein atoms around a position of crystal water, float
          
    output: a tuple of (atoms_info, coordinates, distances) 
    """
    d = distance_array(water_position, heavy_atoms.atoms.positions, result=None, backend='OpenMP', box=box)
    dists = np.sort(d[0])[:N_max]
    indexes = np.argsort(d[0])[:N_max]
    within_cutoff = dists < cutoff
    if np.sum(within_cutoff) >= N_min: 
        indexes = indexes[within_cutoff]
        dists = dists[within_cutoff]
    else:
        dists = np.sort(d[0])[:N_min]
        indexes = np.argsort(d[0])[:N_min] 
    sel_atoms = heavy_atoms.atoms[indexes]
    coords = sel_atoms.atoms.positions
    atoms = [(atom.index, atom.resid, atom.name) for atom in sel_atoms]
    assert len(dists) == len(coords) == len(atoms), "Atom number"
    return atoms, coords, dists

def get_chain_id(atom):
    """
    Get a chain id given an atom index, useful when you have a protein crystal
    """
    return atom.index // N_atoms_in_chain

def find_chains(N_chains=4, N_atoms_in_chain=1473):
    """
    Create a MDAnalysis selection for each individual chain
    """
    chains = [] 
    for i in range(N_chains):
        start =  i * N_atoms_in_chain
        end = start + N_atoms_in_chain - 1
        chain_i = ' and index {}:{}'.format(start, end)
        chains.append(chain_i)
    return chains


def create_connectors(crystal_waters_4, crystal_heavy_atoms_4, box):
    connectors = []
    for cw_oxygens, chain_heavy_atoms in zip(crystal_waters_4, crystal_heavy_atoms_4):
        d = distance_array(
            cw_oxygens.atoms.positions,
            chain_heavy_atoms.atoms.positions,
            box=box
        )
        indexes = np.argmin(d, axis=1)
        connectors.append(chain_heavy_atoms.atoms[indexes].ix)
    return connectors
    
def create_pairwise_bonds(indexes):
    return list(combinations(indexes, 2))

def create_bonds(indexes):
    bonds = []
    for i, index1 in enumerate(indexes):
        if i != len(indexes) - 1:
            bonds.append((index1, indexes[i+1]))
        else:
            bonds.append((index1, indexes[0]))
    return bonds

def create_bonds_w_protein(heavy_atoms_sel_copy, chain_waters_copy, chain_protein_copy):
    bonds = []
    connecting_atoms_list = []
    print(len(heavy_atoms_sel_copy), chain_waters_copy.n_atoms)
    
    for i, (n_heavy_atoms, chain_water) in enumerate(zip(heavy_atoms_sel_copy, chain_waters_copy)):
        connecting_atoms = n_heavy_atoms & chain_protein_copy
        if i == 91 and not connecting_atoms.n_atoms:
            print(i, len(connecting_atoms.atoms), connecting_atoms.atoms, n_heavy_atoms, chain_water)
            connecting_atoms = chain_protein_copy.select_atoms('index 120')
        for atom in connecting_atoms.atoms[:1]:
            bonds.append((atom.ix, chain_water.ix))
        connecting_atoms_list.extend(connecting_atoms.atoms[:1]) 
    return bonds, connecting_atoms_list


def find_pbc_coords(positions, tric_vectors, include_self=False):
  """
  Find all periodic images for the given coordinates in 26 nearest neighbour unit cells
  """
    a, b, c = tric_vectors[0], tric_vectors[1], tric_vectors[2]
    klm = [-1, 0, 1]
    pbc_coords = []
    for k in klm:
        for l in klm:
            for m in klm:
                if not (m==l==k==0):
                    pbc_coords.append(positions + k*a + l*b + m*c)
                elif include_self:
                    pbc_coords.append(positions)
    return np.array(pbc_coords, dtype=np.float32)

def apply_correction(heavy_atoms_sel, box):
    """Function which modifies heavy atoms coordinates according to PBC
    """
    tric_vectors = triclinic_vectors(box)
    coords = []
    
    for i, n_heavy_atoms in enumerate(heavy_atoms_sel):
        
        #mod_coords = apply_PBC(n_heavy_atoms.atoms.positions, box)
        mod_coords = n_heavy_atoms.atoms.positions.copy()
        
        d_pbc = squareform(self_distance_array(
            mod_coords,
            box=box
        ))
        d_cur = squareform(self_distance_array(mod_coords))
        counter = 0
        outlier_list = []
        outlier_index = -1
    
        while np.sum(np.abs(d_pbc - d_cur)) > 29: #  and outlier_index not in outlier_list:
            outlier_index = np.argmax(
                np.sum(np.abs(d_pbc-d_cur), axis=1)
            )
            outlier_list.append(outlier_index)
            pbc_coords = find_pbc_coords(mod_coords[outlier_index], tric_vectors, False)
            others_d = distance_array(pbc_coords, np.delete(mod_coords, outlier_index, axis=0))
            to_chose = np.argmin(np.sum(others_d, axis=1))
            mod_coords[outlier_index] = pbc_coords[to_chose]
            
            d_cur = squareform(self_distance_array(mod_coords))
            d_pbc = squareform(self_distance_array(
                mod_coords,
                box=box,
            ))
            
            counter += 1
            if counter >= 20:
                print(counter, np.sum(np.abs(d_pbc - d_cur)), outlier_index)
                sys.stdout.flush()
                break
                
        coords.append(mod_coords)
    return coords

def visualize_step(crystal_water_positions, connectors, box, chain_waters_save_top):
  """
  This function is useful for recording positions of CWS for each frame. It is possible to visualize them in VMD
  """
    con_pos = np.array(
        [con_atom.position for con_atom in connectors]
    )

    xtc_dist_mtx = distance_array(
        crystal_water_positions,
        con_pos,
        box=box
    )

    tric_vectors = triclinic_vectors(box)

    for m in range(94):
        pbc_ws_coords_m = find_pbc_coords(crystal_water_positions[m], tric_vectors, True)
        d_m = distance_array(
            pbc_ws_coords_m,
            con_pos[m].reshape(-1, 3)
        )

        d_m = d_m.flatten()
        d_m -= xtc_dist_mtx[m,m]
        right_index = np.argmin(np.abs(d_m))
        if np.min(np.abs(d_m)) > 0.1:
            print(d_m)
            sys.stdout.flush()

        chain_waters_save_top.atoms[m].position = pbc_ws_coords_m[right_index]
    
    

def find_offsets(positions_water, positions_crystal, d, box):
    """Function finds offset vectors between the closest water for each CWS"""
    
    dist = np.min(d, axis=1)
    max_d = np.max(dist)
    
    positions_water = apply_PBC(positions_water, box)
    positions_crystal = apply_PBC(positions_crystal, box)
    
    offsets = positions_water - positions_crystal
    norm = np.linalg.norm(offsets, axis=1)
    indeces_to_correct = np.nonzero(np.abs(norm - dist) > 0.2)[0]
    
    if len(indeces_to_correct):

        water_pbc_coords, water_pbc_indeces = augment_coordinates(
            positions_water[indeces_to_correct],
            box,
            100 # to generate periodic images up to 100 angstroms from the position
        )
        crystal_pbc_coords, crystal_pbc_indeces = augment_coordinates(
            positions_crystal[indeces_to_correct],
            box,
            100 # to generate periodic images up to 100 angstroms from the position
        )

        water_all_coords = np.concatenate(
            [positions_water[indeces_to_correct], water_pbc_coords]
        )
        crystal_all_coords = np.concatenate(
            [positions_crystal[indeces_to_correct], crystal_pbc_coords]
        )
        water_all_indeces = np.concatenate(
            [np.arange(len(indeces_to_correct)), water_pbc_indeces]
        )
        crystal_all_indeces = np.concatenate(
            [np.arange(len(indeces_to_correct)), crystal_pbc_indeces]
        )

        for jindex in range(len(indeces_to_correct)):
            subset_water_indexes = water_all_indeces == jindex
            subset_crystal_indexes = crystal_all_indeces == jindex

            subset_water_coords = water_all_coords[subset_water_indexes]
            subset_crystal_coords = crystal_all_coords[subset_crystal_indexes]

            darray = distance_array(subset_water_coords, subset_crystal_coords)

            x, y = np.unravel_index(np.argmin(darray, axis=None), darray.shape)
            if np.min(darray) > max_d + 0.1:
                print(indeces_to_correct[jindex], np.min(darray))
                sys.stdout.flush()

            shift = subset_water_coords[x] - subset_crystal_coords[y] 
            offsets[indeces_to_correct[jindex]] = shift
            
    return offsets
