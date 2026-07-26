"""Tests for the reusable Equation-7 Potts model."""

from __future__ import annotations

import itertools
import unittest

import numpy as np

from tenpy.algorithms.exact_diag import ExactDiag

from potts_model import PottsChain


def _embed_one_site(operator: np.ndarray, site: int, length: int) -> np.ndarray:
    """Embed one local operator in a dense length-``length`` Hilbert space."""

    result = np.array([[1.0]])
    for index in range(length):
        local_operator = operator if index == site else np.eye(3)
        result = np.kron(result, local_operator)
    return result


def _equation_7_dense(length: int, J: float, h: float, periodic: bool) -> np.ndarray:
    """Build a small dense reference Hamiltonian directly from Equation (7)."""

    identity = np.eye(3)
    shifted_state_projectors = []
    for state_index in range(3):
        projector = np.zeros((3, 3))
        projector[state_index, state_index] = 1.0
        shifted_state_projectors.append(projector - identity / 3.0)

    transverse_projector = np.ones((3, 3)) / 3.0 - identity / 3.0
    dimension = 3**length
    hamiltonian = np.zeros((dimension, dimension))

    for site in range(length):
        hamiltonian -= h * _embed_one_site(
            transverse_projector,
            site,
            length,
        )

    bonds = [(site, site + 1) for site in range(length - 1)]
    if periodic:
        bonds.append((length - 1, 0))

    for left_site, right_site in bonds:
        for projector in shifted_state_projectors:
            left_operator = _embed_one_site(projector, left_site, length)
            right_operator = _embed_one_site(projector, right_site, length)
            hamiltonian -= J * (left_operator @ right_operator)

    return hamiltonian


def _dense_model_hamiltonian(model: PottsChain) -> np.ndarray:
    exact_diagonalisation = ExactDiag(model)
    exact_diagonalisation.build_full_H_from_mpo()
    return exact_diagonalisation.full_H.to_ndarray()


def _restricted_basis_indices(
    allowed_states_by_site: list[tuple[int, ...]],
) -> list[int]:
    """Indices of retained product states in the full one-based Potts basis."""

    length = len(allowed_states_by_site)
    indices = []
    for state_tuple in itertools.product(*allowed_states_by_site):
        index = 0
        for state in state_tuple:
            index = 3 * index + state - 1
        indices.append(index)
    return indices


class PottsChainTests(unittest.TestCase):
    def test_registered_operators_are_equation_7_projectors(self):
        model = PottsChain.from_parameters(L=3)
        site = model.lat.unit_cell[0]
        identity = np.eye(3)

        for state in (1, 2, 3):
            q_state = site.get_op(f"Q{state}").to_ndarray()
            p_state = site.get_op(f"P{state}").to_ndarray()

            self.assertTrue(np.allclose(q_state @ q_state, q_state))
            self.assertAlmostEqual(float(np.trace(q_state).real), 1.0)
            self.assertTrue(np.allclose(p_state, q_state - identity / 3.0))
            self.assertAlmostEqual(float(np.trace(p_state).real), 0.0)

        expected_transverse = np.ones((3, 3)) / 3.0 - identity / 3.0
        actual_transverse = site.get_op("Ptransverse").to_ndarray()
        self.assertTrue(np.allclose(actual_transverse, expected_transverse))

    def test_open_hamiltonian_matches_equation_7(self):
        length = 4
        J = 1.2
        h = 0.7
        model = PottsChain.from_parameters(L=length, J=J, h=h)

        expected = _equation_7_dense(length, J, h, periodic=False)
        actual = _dense_model_hamiltonian(model)

        self.assertTrue(np.allclose(actual, expected))

    def test_periodic_hamiltonian_includes_closing_bond(self):
        length = 4
        J = 0.8
        h = 1.1
        model = PottsChain.from_parameters(
            L=length,
            J=J,
            h=h,
            periodic=True,
        )

        expected = _equation_7_dense(length, J, h, periodic=True)
        actual = _dense_model_hamiltonian(model)

        self.assertTrue(model.is_periodic)
        self.assertTrue(np.allclose(actual, expected))

    def test_folded_periodic_order_preserves_the_spectrum(self):
        default_model = PottsChain.from_parameters(
            L=5,
            J=0.9,
            h=1.0,
            periodic=True,
        )
        folded_model = PottsChain.from_parameters(
            L=5,
            J=0.9,
            h=1.0,
            periodic=True,
            order="folded",
        )

        default_energies = np.linalg.eigvalsh(
            _dense_model_hamiltonian(default_model)
        )
        folded_energies = np.linalg.eigvalsh(
            _dense_model_hamiltonian(folded_model)
        )
        self.assertTrue(np.allclose(default_energies, folded_energies))

    def test_fixed_and_forbidden_boundaries_are_exact_restrictions(self):
        length = 3
        free_model = PottsChain.from_parameters(L=length)
        constrained_model = PottsChain.from_parameters(
            L=length,
            left_boundary=1,
            right_boundary=-3,
        )

        self.assertEqual(
            [site.dim for site in constrained_model.lat.mps_sites()],
            [1, 3, 2],
        )
        self.assertEqual(constrained_model.left_allowed_states, (1,))
        self.assertEqual(constrained_model.right_allowed_states, (1, 2))

        full_hamiltonian = _dense_model_hamiltonian(free_model)
        retained = _restricted_basis_indices(
            constrained_model.allowed_states_by_site
        )
        expected = full_hamiltonian[np.ix_(retained, retained)]
        actual = _dense_model_hamiltonian(constrained_model)

        self.assertTrue(np.allclose(actual, expected))

    def test_periodic_chain_rejects_edge_constraints(self):
        with self.assertRaisesRegex(ValueError, "periodic chain has no physical"):
            PottsChain.from_parameters(
                L=4,
                periodic=True,
                left_boundary=1,
            )

    def test_boundary_codes_are_validated(self):
        with self.assertRaisesRegex(ValueError, "left_boundary must be"):
            PottsChain.from_parameters(L=3, left_boundary=4)

    def test_paper_state_numbers_map_to_tenpy_labels(self):
        self.assertEqual(PottsChain.product_state_label(1), "1")
        self.assertEqual(PottsChain.product_state_label(2), "2")
        self.assertEqual(PottsChain.product_state_label(3), "3")

        with self.assertRaisesRegex(ValueError, "state must be 1, 2, or 3"):
            PottsChain.product_state_label(0)

    def test_initial_product_state_respects_restricted_edges(self):
        model = PottsChain.from_parameters(
            L=4,
            left_boundary=2,
            right_boundary=-1,
        )

        self.assertEqual(
            model.initial_product_state(bulk_state=1),
            ["2", "1", "1", "2"],
        )


if __name__ == "__main__":
    unittest.main()
