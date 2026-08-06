# Saved Potts ground states

The DMRG writer notebook stores one self-describing TeNPy MPS ground state per
physical Hamiltonian in this folder. Files are named using the system size,
couplings, physical boundary conditions, and MPS ordering.

The binary files are intentionally ignored by Git because they can be large.
Each file contains:

- the complete MPS needed to inspect Schmidt spectra at arbitrary cuts;
- the physical Hamiltonian parameters;
- DMRG convergence, energy, bond dimension, sweep, tolerance, and runtime
  metadata; and
- a lightweight JSON metadata header that can be indexed without loading the
  MPS tensors.

If another run targets the same Hamiltonian, the writer keeps the converged
state in preference to an unconverged state, then the lower-energy state. When
energies agree at the recorded precision, it keeps the state with the larger
achieved bond dimension.
