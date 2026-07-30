"""Reliable HDF5 storage for finite Potts-chain MPS ground states.

Each file contains the full TeNPy MPS and a JSON-compatible metadata mapping.
The same metadata is duplicated in a small HDF5 root attribute, allowing a
directory to be indexed without loading the large MPS tensors.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
from typing import Any
from uuid import uuid4

import h5py
import numpy as np
from tenpy.tools import hdf5_io


SCHEMA_NAME = "potts_mps_ground_state"
SCHEMA_VERSION = 1
METADATA_ATTRIBUTE = "potts_ground_state_metadata_json"


def _json_ready(value: Any) -> Any:
    """Convert common numerical/container objects to JSON-compatible values."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _normalise_boundary(value: Any) -> int:
    """Return 0 for free, positive for fixed, and negative for forbidden."""

    if value is None:
        return 0
    if isinstance(value, str):
        stripped = value.strip().lower()
        if stripped in {"free", "none", "0"}:
            return 0
        value = int(stripped)
    value = int(value)
    if value not in {-3, -2, -1, 0, 1, 2, 3}:
        raise ValueError(f"Invalid Potts boundary code: {value!r}")
    return value


def _float_token(value: float) -> str:
    """Make a stable, filename-safe token for a scalar coupling."""

    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError("Ground-state filenames require finite scalar J and h.")
    token = format(scalar, ".12g")
    return token.replace("-", "m").replace("+", "p").replace(".", "p")


def _boundary_token(value: Any) -> str:
    code = _normalise_boundary(value)
    if code == 0:
        return "free"
    if code > 0:
        return f"fix{code}"
    return f"forbid{abs(code)}"


def ground_state_filename(metadata: Mapping[str, Any]) -> str:
    """Return the canonical filename for one physical Hamiltonian."""

    physics = metadata["physics"]
    length = int(physics["L"])
    periodic = bool(physics.get("periodic", False))
    order = str(physics.get("order", "default"))
    geometry = "periodic" if periodic else "open"
    left = _boundary_token(physics.get("left_boundary", "free"))
    right = _boundary_token(physics.get("right_boundary", "free"))
    return (
        f"potts_L{length:04d}"
        f"_J{_float_token(physics['J'])}"
        f"_h{_float_token(physics['h'])}"
        f"_{geometry}_L{left}_R{right}_{order}.h5"
    )


def ground_state_path(
    metadata: Mapping[str, Any],
    directory: str | Path,
) -> Path:
    return Path(directory).expanduser().resolve() / ground_state_filename(metadata)


def prepare_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize metadata before it is written."""

    prepared = _json_ready(metadata)
    if "physics" not in prepared or "numerics" not in prepared:
        raise KeyError("Metadata must contain 'physics' and 'numerics' mappings.")

    physics = prepared["physics"]
    physics["L"] = int(physics["L"])
    physics["J"] = float(physics["J"])
    physics["h"] = float(physics["h"])
    physics["left_boundary"] = _normalise_boundary(
        physics.get("left_boundary", "free")
    )
    physics["right_boundary"] = _normalise_boundary(
        physics.get("right_boundary", "free")
    )
    physics["periodic"] = bool(physics.get("periodic", False))
    physics["order"] = str(physics.get("order", "default"))

    prepared["schema_name"] = SCHEMA_NAME
    prepared["schema_version"] = SCHEMA_VERSION
    prepared.setdefault("created_utc", datetime.now(timezone.utc).isoformat())
    return prepared


def read_ground_state_metadata(filename: str | Path) -> dict[str, Any]:
    """Read only the lightweight metadata header, not the MPS tensors."""

    path = Path(filename).expanduser().resolve()
    with h5py.File(path, "r") as handle:
        try:
            encoded = handle.attrs[METADATA_ATTRIBUTE]
        except KeyError as error:
            raise ValueError(
                f"{path.name} is not a recognized Potts ground-state file."
            ) from error
    if isinstance(encoded, bytes):
        encoded = encoded.decode("utf-8")
    metadata = json.loads(str(encoded))
    if metadata.get("schema_name") != SCHEMA_NAME:
        raise ValueError(f"Unexpected ground-state schema in {path.name}.")
    metadata["path"] = str(path)
    metadata["filename"] = path.name
    return metadata


def scan_ground_states(directory: str | Path) -> list[dict[str, Any]]:
    """Index all recognized .h5 files without loading their MPS tensors."""

    folder = Path(directory).expanduser().resolve()
    if not folder.exists():
        return []

    records = []
    for path in sorted(folder.glob("*.h5")):
        try:
            records.append(read_ground_state_metadata(path))
        except (OSError, ValueError, json.JSONDecodeError) as error:
            records.append(
                {
                    "path": str(path),
                    "filename": path.name,
                    "index_error": str(error),
                }
            )
    return records


def load_ground_state(filename: str | Path) -> dict[str, Any]:
    """Load the full MPS payload after the file has been selected."""

    path = Path(filename).expanduser().resolve()
    payload = hdf5_io.load(path)
    if not isinstance(payload, dict) or "psi" not in payload:
        raise ValueError(f"{path.name} does not contain a saved MPS payload.")
    metadata = prepare_metadata(payload["metadata"])
    payload["metadata"] = metadata
    payload["path"] = path
    return payload


def _is_converged(metadata: Mapping[str, Any]) -> bool:
    numerics = metadata.get("numerics", {})
    return bool(
        numerics.get("converged", numerics.get("status") == "converged")
    )


def candidate_is_better(
    candidate: Mapping[str, Any],
    existing: Mapping[str, Any],
) -> tuple[bool, str]:
    """Compare two runs for exactly the same physical Hamiltonian."""

    candidate_converged = _is_converged(candidate)
    existing_converged = _is_converged(existing)
    if candidate_converged != existing_converged:
        if candidate_converged:
            return True, "the new run converged and the stored run did not"
        return False, "the stored run converged and the new run did not"

    candidate_numerics = candidate["numerics"]
    existing_numerics = existing["numerics"]
    candidate_energy = float(candidate_numerics["energy"])
    existing_energy = float(existing_numerics["energy"])
    energy_tolerance = max(1e-11, 1e-12 * max(abs(candidate_energy), abs(existing_energy)))

    if candidate_energy < existing_energy - energy_tolerance:
        return True, "the new run has lower variational energy"
    if candidate_energy > existing_energy + energy_tolerance:
        return False, "the stored run has lower variational energy"

    candidate_chi = int(candidate_numerics.get("chi_max_achieved", 0))
    existing_chi = int(existing_numerics.get("chi_max_achieved", 0))
    if candidate_chi > existing_chi:
        return True, "energies agree and the new run has higher achieved chi"
    if candidate_chi < existing_chi:
        return False, "energies agree and the stored run has higher achieved chi"

    return False, "the stored and new runs are equivalent at the recorded precision"


def save_ground_state(
    psi: Any,
    metadata: Mapping[str, Any],
    directory: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Atomically save an MPS, replacing an existing file only when better."""

    prepared = prepare_metadata(metadata)
    folder = Path(directory).expanduser().resolve()
    folder.mkdir(parents=True, exist_ok=True)
    destination = ground_state_path(prepared, folder)
    destination_existed = destination.exists()

    decision_reason = "no stored state existed"
    if destination_existed and not force:
        existing = read_ground_state_metadata(destination)
        better, decision_reason = candidate_is_better(prepared, existing)
        if not better:
            return {
                "action": "kept_existing",
                "path": destination,
                "reason": decision_reason,
                "metadata": existing,
            }

    temporary = destination.with_name(
        f"{destination.stem}.tmp-{uuid4().hex}{destination.suffix}"
    )
    payload = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "metadata": prepared,
        "psi": psi,
    }

    try:
        hdf5_io.save(payload, temporary)
        with h5py.File(temporary, "a") as handle:
            handle.attrs[METADATA_ATTRIBUTE] = json.dumps(
                prepared,
                sort_keys=True,
                separators=(",", ":"),
            )
            handle.flush()
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()

    return {
        "action": "replaced" if destination_existed else "saved",
        "path": destination,
        "reason": decision_reason,
        "metadata": prepared,
    }
