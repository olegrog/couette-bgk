[![DOI](https://zenodo.org/badge/212660039.svg)](https://zenodo.org/badge/latestdoi/212660039)

# Couette BGK
Solver for the linear Couette-flow problem for a rarefied BGK gas, which is reduced to the 1D integral equation.

The code is based on the algorithm presented in
- S. Jiang, L.-S. Luo, Analysis and accurate numerical solutions of the integral equation derived from the linearized BGKW equation for the steady Couette flow, J. Comput. Phys. 316 (2016) 416–434. doi: [10.1016/j.jcp.2016.04.011](http://doi.org/10.1016/j.jcp.2016.04.011).

Example of usage:
```
./exact-bgkw.py -k=0.1 -Nu=256 -Ng=4
```
The output consists of 4 columns: coordinate `X`, velocity `v_x`, shear stress `p_xy`, longitudinal heat flow `q_x`.
