# LAWS: Local Alignment for Water Sites
## A method to analyze crystallographic water in simulations 

<img align ="center" src="Fig2.png" width="600">

## Description
- `laws.py` contains the multilateration algorithm and all the necessary optimization pipelines
- `compute_cws_r.py` contains the example of application of the LAWS to the real system
- `compute_bulk_r.py` contains the example of generating offset vectors for bulk water sites (in a real system)


## Installation

The LAWS algorithm relies on the open-source `MDAnalysis` module [MDAanalysis](https://www.mdanalysis.org/) and `scipy` optimization, as well as `MPI4py` library for the parallel computing: 

```
pip install scipy
pip install --upgrade MDAnalysis
pip install mpi4py
```

## Standard Command Line Usage (with MPI)
Example:
```
mpirun -np 4 python compute_cws_r.py folder_with_data cws_filename folder_to_save_vis
mpirun -np 4 python compute_bulk_r.py folder_with_data bulk_filename
```
## MD simulation data
MD simulation data is publicly available from [Zenodo](https://doi.org/10.5281/zenodo.6478270) at DOI: https://doi.org/10.5281/zenodo.6478270

## Citation
If you use these codes in your research, please cite our manuscript:

```
@software{LAWS,
  author       = {Klyshko, Eugene and Kim, Justin and Rauscher, Sarah},
  title        = {{LAWS: Local Alignment for Water Sites - a method to analyze crystallographic water in simulations }},
  url          = {https://github.com/rauscher-lab/LAWS/}
}
```
