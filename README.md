# LAWS: Local Alignment for Water Sites
## Algorithm for tracking ordered water in simulations 

Manuscript: https://doi.org/10.1016/j.bpj.2022.09.012

<img align ="center" src="Fig2.png" width="600">

## Abstract

Accurate modeling of protein-water interactions in molecular dynamics (MD) simulations is important for understanding the molecular basis of protein function. Data from x-ray crystallography can be useful in assessing the accuracy of MD simulations, in particular, the locations of crystallographic water sites (CWS) coordinated by the protein. Such a comparison requires special methodological considerations that take into account the dynamic nature of proteins. However, existing methods for analyzing CWS in MD simulations rely on global alignment of the protein onto the crystal structure, which introduces substantial errors in the case of significant structural deviations. Here, we propose a method called local alignment for water sites (LAWS), which is based on multilateration—an algorithm widely used in GPS tracking. LAWS considers the contacts formed by CWS and protein atoms in the crystal structure and uses these interaction distances to track CWS in a simulation. We apply our method to simulations of a protein crystal and to simulations of the same protein in solution. Compared with existing methods, LAWS defines CWS characterized by more prominent water density peaks and a less-perturbed protein environment. In the crystal, we find that all high-confidence crystallographic waters are preserved. Using LAWS, we demonstrate the importance of crystal packing for the stability of CWS in the unit cell. Simulations of the protein in solution and in the crystal share a common set of preserved CWS that are located in pockets and coordinated by residues of the same domain, which suggests that the LAWS algorithm will also be useful in studying ordered waters and water networks in general.

## Description
- `laws.py` contains the multilateration algorithm and all the necessary optimization pipelines.
- `compute_bulk_r.py` contains the code to compute offset vectors **r** and distances |r| for bulk water sites.
- `compute_global_r.py` contains the code to compute offset vectors **r**, distances |r| for globally aligned CWS.
- `compute_laws_r.py` contains the code to compute offset vectors **r**, distances |r| and the correponding LAWS errors for the CWS tracked by our local alignment algorithm (LAWS). Additionally, it saves moving positions of the CWS into a new trajectory which can be visualized (as in Fig.4A of the manuscript) with VMD or PyMOL.
- `crystal.gro` contains the constructed unit cell of the system described in the manuscript (PDZ domain of a human LNX2), with a total of four symmetrically related chains and 4 copies of 94 CWS. Note that the protein atoms numbering corresponds to the numbering in simulation file `firsframe_fix.gro` (available at https://doi.org/10.5281/zenodo.6478270). CWS are listed consequtively after all protein atoms of four chains and have resids 501-594. 


## Installation

The LAWS algorithm relies on the open-source `MDAnalysis` [module](https://www.mdanalysis.org/) and `scipy` optimization, as well as `MPI4py` library for the parallel computing: 

```
pip install scipy
pip install --upgrade MDAnalysis
pip install mpi4py
```

## Standard Command Line Usage (with MPI)

User should adjust parameters in the code according to MD system they analyze (see details in comment section of each file): 
- `firstframe.gro` - simulation structure file (.GRO or similar).
- `trajectory.xtc` - simulation trajectory file (.XTC or similar).
- `crystal.gro` - crystal structure (protein + crystal waters).
- `N_chains`- number of symmetric chains in the simulation.
- `N_atoms_in_chain` - number of protein atoms in each chain.
- `n_waters` - number of CWS in the crystal structure

After adjusting parameters specific to your MD simulation, run the codes:
```
python compute_bulk_r.py
mpirun -np 4 python compute_laws_r.py
mpirun -np 4 python compute_global_r.py
```
The LAWS algorithm requires solving a nonlinear optimization problem for each CWS at every frame. We implemented the MPI protocol to parallelize and speed up the computations. For the system described in the manuscript (4 protein chains x 94 CWS x 100,000 frames), it took 5 hours of wall clock time on a node with 2 Intel® Xeon® Gold 6148 Processors (2.4 GHz) to obtain the results. We used `mpirun -np 40` to run the code with 40 MPI processes. 

The output of these programs `distances.npy` can be used for plotting distribution $P(r)$ and radial density function $g(r) = P(r) / r^2$, or `offsets.npy` as an input for 3D density calculators (e.g. Gromaps). 

## MD simulation data
MD simulation data is publicly available from [Zenodo](https://doi.org/10.5281/zenodo.6478270) at DOI: https://doi.org/10.5281/zenodo.6478270. These structure and trajectory files can be used to test the algorithm. The constructed unit cell of a crystal structure (second PDZ domain of LNX2, PDBID: 5E11) with CWS is available from here: [crystal.gro](https://github.com/rauscher-lab/LAWS/blob/main/crystal.gro).

## Citation
If you use these codes in your research, please cite our manuscript:
```
@article{LAWS2022,
	title = {LAWS: Local alignment for water sites – Tracking ordered water in simulations},
	journal = {Biophysical Journal},
	year = {2022},
	issn = {0006-3495},
	doi = {https://doi.org/10.1016/j.bpj.2022.09.012},
	url = {https://www.sciencedirect.com/science/article/pii/S0006349522007627},
	author = {Eugene Klyshko and Justin Sung-Ho Kim and Sarah Rauscher},
}

```
