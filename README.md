# Three-State Potts Model

This project contains a reusable TeNPy implementation of the three-state
quantum Potts model, together with tests and example DMRG notebooks.

The model implements

```text
H = -J sum_(<i,j>) sum_(mu=1)^3 P_i^mu P_j^mu - h sum_i P_i
```

with

```text
P^mu = |mu><mu| - I/3
P    = |lambda_0><lambda_0| - I/3
|lambda_0> = (|1> + |2> + |3>) / sqrt(3)
```

The paper's `J`, `h`, and one-based state labels are used directly.

## Example notebooks

- `notebooks/simple_potts_boundary_comparison.ipynb` provides a complete
  boundary-comparison workflow. It constructs fixed, forbidden, and free edge
  conditions, runs finite DMRG, checks the exact edge probabilities, plots
  state-probability and entanglement-entropy profiles, and performs rough open-
  and periodic-chain CFT fits.
- `notebooks/potts_entanglement_spectrum.ipynb` calculates the half-chain
  entanglement spectrum of the critical model for several finite system sizes.
  Its default comparison uses a free-boundary open chain and a periodic chain.
  It validates the Schmidt-spectrum normalization, shows the raw shifted
  spectra, and produces a Figure-5-style finite-size spectrum-flow plot against
  `1 / log(L)`. The notebook does not extrapolate thermodynamic-limit levels;
  quantitatively converging the higher spectrum requires substantially larger
  bond dimensions and computing resources.
- `notebooks/potts_generate_ground_states.ipynb` is the expensive half of the
  saved-state workflow. It runs DMRG and immediately stores one self-describing
  HDF5 MPS file per physical Hamiltonian in `data/ground_states/`. Both odd and
  even system sizes are supported; odd-length states are checked at both bonds
  adjacent to the half-integer centre.
- `notebooks/potts_entanglement_spectrum_saved.ipynb` is the inexpensive half.
  It indexes saved metadata, loads only matching MPS files, and reproduces the
  raw, scaled, and cut-residue entanglement-spectrum plots without running
  DMRG. Total-size parity and cut parity are separated into four sectors.

## Saved-ground-state workflow

Run `potts_generate_ground_states.ipynb` when new or more accurate ground
states are needed. A sufficiently accurate converged state is skipped before
DMRG, and each accepted state is saved immediately after its run. The writer
prefers converged states, then lower variational energy, then larger achieved
bond dimension.

The HDF5 filenames contain the system size, couplings, physical boundary
conditions, and MPS ordering. Numerical details such as requested and achieved
bond dimension, sweep count, convergence status, tolerances, runtime, energy,
and canonical-form error are stored inside each file. A lightweight metadata
header lets the plotting notebook identify matching files without loading all
MPS tensors.

The binary ground-state files are ignored by Git because they can be large.
The explanatory `data/ground_states/README.md` remains tracked.

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Launch Jupyter from the project directory:

```bash
jupyter lab
```

## Using the model

When running a script from the project directory, make the package importable:

```bash
export PYTHONPATH="$PWD"
```

Create an open chain with independent edge conditions:

```python
from potts_model import PottsChain

model = PottsChain.from_parameters(
    L=30,
    J=1.0,
    h=1.0,
    left_boundary=1,       # fix paper state 1
    right_boundary=-3,     # forbid paper state 3
)

initial_state = model.initial_product_state(bulk_state=1)
```

Supported edge codes:

| Code | Meaning |
| --- | --- |
| `1`, `2`, `3` | Fix that paper state |
| `-1`, `-2`, `-3` | Forbid that paper state |
| `"free"` or `None` | Retain all three states |

The edge conditions are exact local Hilbert-space restrictions. A fixed edge
retains one state, while a forbidden edge retains the other two.

Create a periodic ring:

```python
ring = PottsChain.from_parameters(
    L=30,
    J=1.0,
    h=1.0,
    periodic=True,
    order="folded",
)
```

A periodic chain has no edges, so its left and right conditions must both be
free. The folded ordering is recommended for finite-system DMRG because it
keeps the physical nearest-neighbour interactions short-ranged in MPS order.

## TeNPy DMRG option for open chains longer than 18 sites

Exact fixed and forbidden edges require different local dimensions at the
boundaries. The model therefore uses a TeNPy `TrivialLattice`, whose internal
unit cell contains the full finite chain. For `L > 18`, set the following DMRG
option so TeNPy's generic higher-dimensional safety check does not mistake the
chain for a wide cylinder:

```python
dmrg_params = {
    "max_N_sites_per_ring": model.lat.N_sites,
    # Other DMRG options...
}
```

This only overrides that geometry heuristic; it does not change the
Hamiltonian or the DMRG calculation.

## Running the tests

From the project directory with the virtual environment activated:

```bash
PYTHONPATH=. python -m unittest discover -s tests -v
```
