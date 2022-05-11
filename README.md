# LAWS: Local Alignment for Water Sites
## Tracking ordered water in simulations 

<img align ="center" src="Fig2.png" width="600">

## Description
- `laws.py` contains the multilateration algorithm and all the necessary optimization pipelines.
- `compute_bulk_r.py` contains the code to compute offset vectors **r** and distances |r| for bulk water sites.
- `compute_global_r.py` contains the code to compute offset vectors **r**, distances |r| for globally aligned CWS.
- `compute_laws_r.py` contains the code to compute offset vectors **r**, distances |r| and the correponding LAWS errors for the CWS tracked by our local alignment algorithm (LAWS). Additionally, it saves moving positions of the CWS into new trajectory which can be visualized (as in Fig.4A of the manuscript)


## Installation

The LAWS algorithm relies on the open-source `MDAnalysis` module [MDAanalysis](https://www.mdanalysis.org/) and `scipy` optimization, as well as `MPI4py` library for the parallel computing: 

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
- `n_waters` = 94 - number of CWS in the crystal structure

After adjusting parameters specific to your MD simulation, run the codes:
```
python compute_bulk_r.py
mpirun -np 4 python compute_laws_r.py
mpirun -np 4 python compute_global_r.py
```

## MD simulation data
MD simulation data is publicly available from [Zenodo](https://doi.org/10.5281/zenodo.6478270) at DOI: https://doi.org/10.5281/zenodo.6478270. Can be used to test the algorithm. 

## Citation
If you use these codes in your research, please cite our manuscript:

```
@software{LAWS,
  author       = {Klyshko, Eugene and Kim, Justin and Rauscher, Sarah},
  title        = {{LAWS: Local Alignment for Water Sites - a method to analyze crystallographic water in simulations }},
  url          = {https://github.com/rauscher-lab/LAWS/}
}
```
