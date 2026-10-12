"""Resolve a registered local fit without losing the original deletion lineage."""
from pathlib import Path

from .workspace import ExistingSandError, _load_yaml, _safe_relative, _stage_location


def fitted_checkpoint(run: Path, fit_id: str, derived_id: str):
    stage = _stage_location(run, 'background', fit_id)
    if not (stage / 'manifest.yaml').is_file():
        raise ExistingSandError(f'unregistered E3 fit: {stage}')
    record = _load_yaml(stage / 'record.yaml', 'E3 fit')
    if record.get('prompt', {}).get('type') != 'pgsr_local_background_optimization':
        raise ExistingSandError('source_fit_id must identify an E3 local optimization')
    params = record['parameters']
    if params['derived_id'] != derived_id:
        raise ExistingSandError('source fit and completion have different derived checkpoints')
    relative, cloud = _safe_relative(stage / 'checkpoint', params['checkpoint_point_cloud'], 'fitted cloud')
    registered = {item['path'] for item in _load_yaml(stage / 'manifest.yaml', 'fit manifest')['artifacts']}
    if not cloud.is_file() or f'checkpoint/{relative}' not in registered:
        raise ExistingSandError('fitted cloud is not a registered checkpoint artifact')
    return stage, relative, cloud, record
