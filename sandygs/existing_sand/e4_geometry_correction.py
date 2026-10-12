"""Auditable E4 tabletop position proposals for selected E3 Gaussians."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from .workspace import ExistingSandError, _sha256


def tabletop_proposal(vertices, candidate_indices, terrain, grid, *, first_new,
                      last_new, neighborhood_radius=0.02):
    """Return a conservative XY-fixed proposal and a local trainable set.

    The E4 field is cell centered and indexed [x,y]. The caller verifies the
    exchange transform and source cloud before using this proposal.
    """
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if (candidate_indices.ndim != 1 or not len(candidate_indices)
            or len(np.unique(candidate_indices)) != len(candidate_indices)
            or np.any(candidate_indices < first_new)
            or np.any(candidate_indices >= last_new)):
        raise ExistingSandError('E4 position candidates must be unique 55-group indices')
    if not np.isfinite(neighborhood_radius) or neighborhood_radius < 0:
        raise ExistingSandError('neighborhood radius must be finite nonnegative')
    xyz = np.column_stack([vertices[k] for k in ('x', 'y', 'z')]).astype(np.float64)
    selected = xyz[candidate_indices]
    origin = np.asarray(grid['cell_center_origin'], dtype=np.float64)
    spacing = np.asarray(grid['spacing'], dtype=np.float64)
    ij = (selected[:, :2] - origin) / spacing
    if (terrain.ndim != 2 or np.any(~np.isfinite(terrain))
            or np.any(ij[:, 0] < 0) or np.any(ij[:, 0] > terrain.shape[0] - 1)
            or np.any(ij[:, 1] < 0) or np.any(ij[:, 1] > terrain.shape[1] - 1)):
        raise ExistingSandError('candidate is outside valid E4 tabletop cells')
    target_z = map_coordinates(terrain, [ij[:, 0], ij[:, 1]], order=1,
                               mode='nearest').astype(np.float32)
    neighbors = cKDTree(xyz[first_new:last_new, :2]).query_ball_point(
        selected[:, :2], r=neighborhood_radius)
    train_indices = np.unique(np.concatenate(
        [candidate_indices] + [np.asarray(group, dtype=np.int64) + first_new
                               for group in neighbors]))
    return target_z, train_indices


def load_position_proposal(path: Path, source_cloud: Path, *, old_count: int,
                           point_count: int):
    """Validate a proposal against the exact immutable E3 source cloud."""
    with np.load(path, allow_pickle=False) as data:
        required = {'indices', 'target_z', 'train_indices', 'source_sha256'}
        if set(data.files) != required:
            raise ExistingSandError('position proposal has unexpected fields')
        indices = np.asarray(data['indices'], dtype=np.int64)
        target_z = np.asarray(data['target_z'], dtype=np.float32)
        train_indices = np.asarray(data['train_indices'], dtype=np.int64)
        source_digest = str(data['source_sha256'].item())
    if source_digest != _sha256(source_cloud):
        raise ExistingSandError('position proposal source PLY hash mismatch')
    for name, values in (('indices', indices), ('train_indices', train_indices)):
        if (values.ndim != 1 or not len(values) or len(np.unique(values)) != len(values)
                or np.any(values < old_count) or np.any(values >= point_count)):
            raise ExistingSandError(f'position proposal {name} are invalid')
    if target_z.shape != indices.shape or not np.isfinite(target_z).all():
        raise ExistingSandError('position proposal target_z is invalid')
    if not np.isin(indices, train_indices).all():
        raise ExistingSandError('position proposal targets must be trainable')
    return indices, target_z, train_indices
