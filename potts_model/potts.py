r"""Reusable three-state quantum Potts chain in the convention of Chepiga and Mila.

This module implements Equation (7) of

    N. Chepiga and F. Mila, *Excitation spectrum and Density Matrix
    Renormalization Group iterations* (2017).

The Hamiltonian is

.. math::

    H = -J \sum_{\langle i,j\rangle}\sum_{\mu=1}^{3}
        P_i^\mu P_j^\mu
        -h\sum_i P_i,

where

.. math::

    P^\mu = |\mu\rangle\langle\mu| - \frac{I}{3},\qquad
    P = |\lambda_0\rangle\langle\lambda_0| - \frac{I}{3},\qquad
    |\lambda_0\rangle =
        \frac{|1\rangle+|2\rangle+|3\rangle}{\sqrt{3}}.

"""

from __future__ import annotations

from numbers import Integral
from typing import Any, TypeAlias

import numpy as np

from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Chain, TrivialLattice
from tenpy.models.model import CouplingMPOModel
from tenpy.networks.site import Site


BoundaryCode: TypeAlias = int | str | None

POTTS_STATES = (1, 2, 3)
_VALID_NONFREE_BOUNDARY_CODES = {-3, -2, -1, 1, 2, 3}


def _normalise_boundary_code(value: BoundaryCode, name: str) -> int:
    """Return the canonical integer representation of one edge condition."""

    if isinstance(value, str):
        value = value.strip().lower()

    if value is None or value == "free" or (
        isinstance(value, Integral) and not isinstance(value, bool) and int(value) == 0
    ):
        return 0

    if isinstance(value, Integral) and not isinstance(value, bool):
        code = int(value)
        if code in _VALID_NONFREE_BOUNDARY_CODES:
            return code

    raise ValueError(
        f"{name} must be 'free', None, 1, 2, 3, -1, -2, or -3; "
        f"received {value!r}."
    )


def _allowed_states(code: int) -> tuple[int, ...]:
    """Translate a boundary code into the states retained at that edge."""

    if code == 0:
        return POTTS_STATES
    if code > 0:
        return (code,)
    forbidden_state = abs(code)
    return tuple(state for state in POTTS_STATES if state != forbidden_state)


def _potts_site(allowed_states: tuple[int, ...]) -> Site:
    """Create a local site by restricting Equation-7 operators to allowed states.

    Interior and periodic sites retain all three states. At an open edge, a
    fixed boundary gives a one-dimensional site and a forbidden boundary gives
    a two-dimensional site. This makes the boundary condition an exact Hilbert
    space restriction rather than a large but finite boundary field.
    """

    if not allowed_states or any(state not in POTTS_STATES for state in allowed_states):
        raise ValueError(f"invalid allowed Potts states: {allowed_states!r}")

    full_identity = np.eye(3, dtype=float)
    full_operators: dict[str, np.ndarray] = {}

    for state in POTTS_STATES:
        projector = np.zeros((3, 3), dtype=float)
        projector[state - 1, state - 1] = 1.0
        full_operators[f"Q{state}"] = projector
        full_operators[f"P{state}"] = projector - full_identity / 3.0

    lambda_projector = np.ones((3, 3), dtype=float) / 3.0
    full_operators["Qlambda0"] = lambda_projector
    full_operators["Ptransverse"] = lambda_projector - full_identity / 3.0

    retained_indices = np.asarray(allowed_states, dtype=int) - 1
    leg = npc.LegCharge.from_trivial(len(allowed_states))
    site = Site(
        leg,
        state_labels=[str(state) for state in allowed_states],
        sort_charge=False,
    )
    neutral_charge = leg.chinfo.make_valid()

    for name, full_operator in full_operators.items():
        restricted = full_operator[np.ix_(retained_indices, retained_indices)]
        operator = npc.Array.from_ndarray(
            restricted,
            [leg, leg.conj()],
            qtotal=neutral_charge,
        )
        site.add_op(name, operator, hc=name)

    return site


def _parameter_values(value: Any, count: int, name: str) -> np.ndarray:
    """Broadcast a scalar or validate a one-dimensional spatial parameter."""

    values = np.asarray(value, dtype=float)
    if values.ndim == 0:
        return np.full(count, float(values))
    if values.shape == (count,):
        return values
    raise ValueError(
        f"{name} must be a real scalar or an array of shape ({count},); "
        f"received shape {values.shape}."
    )


class PottsChain(CouplingMPOModel):
    r"""Finite three-state Potts chain using Equation (7) literally.

    Use :meth:`from_parameters` for the common scalar-parameter interface, or
    pass a normal TeNPy parameter dictionary to ``PottsChain({...})``.

    Model parameters
    ----------------
    L : int
        Number of sites. Must be at least ``2``.
    J : real scalar or array
        Equation-7 nearest-neighbour coupling. For an open chain an array must
        have shape ``(L - 1,)``; for a periodic chain it must have shape
        ``(L,)``. The default is ``1.0``.
    h : real scalar or array
        Equation-7 transverse coupling. An array must have shape ``(L,)``.
        The default is ``1.0``. The ferromagnetic model in the paper is
        critical at ``h == J``.
    left_boundary, right_boundary : {"free", None, 1, 2, 3, -1, -2, -3}
        Independently specifies each edge of an open chain:

        - ``1``, ``2``, or ``3`` retains only that state (fixed).
        - ``-1``, ``-2``, or ``-3`` removes that state and retains the other
          two (forbidden, or partially fixed).
        - ``"free"`` or ``None`` retains all three states.

        These are exact local Hilbert-space restrictions; no boundary field or
        penalty-strength limit is used.
    bc_x : {"open", "periodic"}
        Physical boundary condition. ``"periodic"`` adds the closing bond
        between physical sites ``L - 1`` and ``0`` and requires both edge
        conditions to be free.
    order : {"default", "folded"}
        MPS site ordering. ``"folded"`` is supported for periodic rings and
        makes all physical nearest-neighbour bonds short-ranged in the MPS.
        Exact open-edge restrictions currently require ``"default"``.
    bc_MPS : {"finite"}
        This class currently represents finite systems. A periodic Hamiltonian
        still uses a finite open-boundary MPS.
    conserve : None
        The direct projector-basis implementation currently requires ``None``.

    Registered local operators
    --------------------------
    ``Q1``, ``Q2``, ``Q3``
        Ordinary projectors :math:`|\mu\rangle\langle\mu|`. Their expectation
        values are the probabilities of the three paper states.
    ``P1``, ``P2``, ``P3``
        Shifted state projectors :math:`P^\mu` from Equation (7).
    ``Qlambda0``
        Projector :math:`|\lambda_0\rangle\langle\lambda_0|`.
    ``Ptransverse``
        Shifted transverse projector :math:`P` from Equation (7).

    Notes
    -----
    State labels are one-based everywhere in this class, matching the paper.
    For example, a uniform state-1 product state is ``["1"] * L``. The helper
    :meth:`initial_product_state` automatically chooses an allowed state when
    the requested bulk state is excluded at an edge.

    A periodic Hamiltonian represented with a finite MPS has a closing bond.
    With ``order="default"`` that bond spans the full MPS. The recommended
    ``order="folded"`` distributes the cost into short next-nearest-neighbour
    interactions.

    Examples
    --------
    Fix the left edge to state 1 and forbid state 3 at the right edge:

    >>> model = PottsChain.from_parameters(
    ...     L=30,
    ...     J=1.0,
    ...     h=1.0,
    ...     left_boundary=1,
    ...     right_boundary=-3,
    ... )
    >>> model.left_allowed_states
    (1,)
    >>> model.right_allowed_states
    (1, 2)

    Construct a periodic ring:

    >>> ring = PottsChain.from_parameters(
    ...     L=30,
    ...     J=1.0,
    ...     h=1.0,
    ...     periodic=True,
    ...     order="folded",
    ... )
    """

    @classmethod
    def from_parameters(
        cls,
        L: int,
        J: float = 1.0,
        h: float = 1.0,
        *,
        left_boundary: BoundaryCode = "free",
        right_boundary: BoundaryCode = "free",
        periodic: bool = False,
        order: str = "default",
        **model_params: Any,
    ) -> "PottsChain":
        """Construct a finite chain without manually assembling a dictionary."""

        if "bc_x" in model_params:
            raise ValueError(
                "from_parameters() sets bc_x from periodic; pass periodic=True "
                "instead of supplying bc_x."
            )
        if "bc_MPS" in model_params and model_params["bc_MPS"] != "finite":
            raise ValueError(
                "PottsChain currently supports finite systems only; "
                "bc_MPS must be 'finite'."
            )

        params = {
            "L": L,
            "J": J,
            "h": h,
            "left_boundary": left_boundary,
            "right_boundary": right_boundary,
            "bc_MPS": "finite",
            "bc_x": "periodic" if periodic else "open",
            "order": order,
            "conserve": None,
        }
        params.update(model_params)
        return cls(params)

    @staticmethod
    def product_state_label(state: int) -> str:
        """Return the TeNPy label for paper state 1, 2, or 3."""

        if isinstance(state, bool) or not isinstance(state, Integral) or int(state) not in POTTS_STATES:
            raise ValueError(f"state must be 1, 2, or 3; received {state!r}.")
        return str(int(state))

    def initial_product_state(self, bulk_state: int = 1) -> list[str]:
        """Return a valid product state in the model's actual MPS ordering.

        ``bulk_state`` is used wherever it is allowed. At a boundary that
        excludes it, the lowest-numbered allowed state is selected instead.
        """

        self.product_state_label(bulk_state)
        labels = []
        for physical_site in self._mps_to_physical:
            allowed = self.allowed_states_by_site[physical_site]
            selected = bulk_state if bulk_state in allowed else allowed[0]
            labels.append(str(selected))
        return labels

    def init_lattice(self, model_params):
        """Build full interior sites and exactly restricted boundary sites."""

        # These values are used later in init_terms(). Mark them as recognized
        # now so an earlier boundary-validation error does not produce a
        # misleading TeNPy "unused option" warning.
        model_params.touch("J", "h")

        length = model_params.get("L", 2, int)
        if length < 2:
            raise ValueError(f"L must be at least 2; received L={length}.")

        q = model_params.get("q", 3, int)
        if q != 3:
            raise ValueError(
                "Equation (7) is the three-state Potts model, so q must be 3; "
                f"received q={q}."
            )

        conserve = model_params.get("conserve", None)
        if conserve is not None:
            raise ValueError(
                "PottsChain encodes Equation (7) directly in the |1>, |2>, "
                "|3> basis and currently requires conserve=None."
            )

        bc_mps = model_params.get("bc_MPS", "finite", str)
        if bc_mps != "finite":
            raise ValueError(
                "PottsChain currently supports finite systems only; "
                f"received bc_MPS={bc_mps!r}."
            )

        bc_x = model_params.get("bc_x", "open", str).lower()
        if bc_x not in ("open", "periodic"):
            raise ValueError(
                "bc_x must be 'open' or 'periodic'; "
                f"received {bc_x!r}."
            )

        order = model_params.get("order", "default", str)
        left_value = model_params.get("left_boundary", "free")
        right_value = model_params.get("right_boundary", "free")
        left = _normalise_boundary_code(left_value, "left_boundary")
        right = _normalise_boundary_code(right_value, "right_boundary")

        self.left_boundary = left
        self.right_boundary = right
        self.left_allowed_states = _allowed_states(left)
        self.right_allowed_states = _allowed_states(right)
        self.is_periodic = bc_x == "periodic"

        if self.is_periodic and (left != 0 or right != 0):
            raise ValueError(
                "A periodic chain has no physical left or right edge. Set both "
                "left_boundary and right_boundary to 'free'."
            )

        if self.is_periodic:
            full_site = _potts_site(POTTS_STATES)
            lattice = Chain(
                length,
                full_site,
                bc="periodic",
                bc_MPS="finite",
                order=order,
            )
            self.allowed_states_by_site = [POTTS_STATES] * length
            self._mps_to_physical = [int(index) for index in lattice.order[:, 0]]
        else:
            if order != "default":
                raise ValueError(
                    "Open chains with exact edge restrictions require "
                    "order='default'."
                )

            self.allowed_states_by_site = [POTTS_STATES] * length
            self.allowed_states_by_site[0] = self.left_allowed_states
            self.allowed_states_by_site[-1] = self.right_allowed_states
            sites = [
                _potts_site(allowed)
                for allowed in self.allowed_states_by_site
            ]
            lattice = TrivialLattice(sites)
            self._mps_to_physical = list(range(length))

        self._physical_to_mps = np.empty(length, dtype=int)
        for mps_index, physical_index in enumerate(self._mps_to_physical):
            self._physical_to_mps[physical_index] = mps_index

        return lattice

    def init_sites(self, model_params):
        """Unused because :meth:`init_lattice` creates site-specific edges."""

        raise RuntimeError("PottsChain.init_lattice() creates its sites directly.")

    def init_terms(self, model_params):
        """Add the onsite and bond terms of Equation (7)."""

        length = self.lat.N_sites
        h_values = _parameter_values(
            model_params.get("h", 1.0, "real_or_array"),
            length,
            "h",
        )
        number_of_bonds = length if self.is_periodic else length - 1
        J_values = _parameter_values(
            model_params.get("J", 1.0, "real_or_array"),
            number_of_bonds,
            "J",
        )

        for physical_site, h_value in enumerate(h_values):
            mps_site = int(self._physical_to_mps[physical_site])
            self.add_onsite_term(
                -h_value,
                mps_site,
                "Ptransverse",
                category=f"transverse_projector_site_{physical_site}",
            )

        for physical_left, J_value in enumerate(J_values):
            physical_right = (physical_left + 1) % length
            mps_left = int(self._physical_to_mps[physical_left])
            mps_right = int(self._physical_to_mps[physical_right])

            # add_coupling_term requires increasing MPS indices. P^mu is the
            # same Hermitian operator on both endpoints, so swapping endpoints
            # leaves the physical term unchanged.
            first, second = sorted((mps_left, mps_right))
            for state in POTTS_STATES:
                self.add_coupling_term(
                    -J_value,
                    first,
                    second,
                    f"P{state}",
                    f"P{state}",
                    category=(
                        f"potts_bond_{physical_left}_{physical_right}"
                        f"_state_{state}"
                    ),
                )


__all__ = ["BoundaryCode", "POTTS_STATES", "PottsChain"]
