from pathlib import Path

import numpy as np
import pytest

from sandygs.existing_sand.e4_geometry_correction import (
    load_position_proposal, tabletop_proposal,
)
from sandygs.existing_sand.workspace import ExistingSandError, _sha256


def test_tabletop_proposal_uses_exchange_xy_cell_centers_and_local_neighbors():
    vertices = np.zeros(4, dtype=[('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
    vertices['x'][:] = [0, 1, 1.1, 3]
    vertices['y'][:] = [0, 1, 1, 3]
    vertices['z'][:] = [0, 5, 6, 7]
    terrain = np.fromfunction(lambda x, y: 10 * x + y, (5, 5)).astype(np.float32)
    target, train = tabletop_proposal(
        vertices, np.array([1]), terrain,
        {'cell_center_origin': [0, 0], 'spacing': [1, 1]},
        first_new=1, last_new=4, neighborhood_radius=.11)
    assert target.tolist() == [11]
    assert train.tolist() == [1, 2]


def test_proposal_rejects_wrong_source_cloud(tmp_path: Path):
    cloud = tmp_path / 'source.ply'
    cloud.write_bytes(b'checkpoint A')
    proposal = tmp_path / 'proposal.npz'
    np.savez(proposal, indices=np.array([2]), target_z=np.array([1.0]),
             train_indices=np.array([2, 3]), source_sha256=np.asarray(_sha256(cloud)))
    indices, z, train = load_position_proposal(proposal, cloud, old_count=2, point_count=4)
    assert (indices.tolist(), z.tolist(), train.tolist()) == ([2], [1.0], [2, 3])
    cloud.write_bytes(b'checkpoint B')
    with pytest.raises(ExistingSandError, match='hash mismatch'):
        load_position_proposal(proposal, cloud, old_count=2, point_count=4)
