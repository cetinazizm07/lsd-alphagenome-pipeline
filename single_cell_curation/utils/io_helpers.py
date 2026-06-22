from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import anndata

log = logging.getLogger(__name__)


def save_checkpoint(ckpt_path: Path, h5ad_path: Path, metadata: dict[str, Any]) -> None:
    payload = {"h5ad_path": str(h5ad_path), **metadata}
    ckpt_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.debug("Checkpoint written: %s", ckpt_path)


def load_checkpoint(ckpt_path: Path) -> tuple[Path, dict[str, Any]]:
    payload = json.loads(ckpt_path.read_text(encoding="utf-8"))
    h5ad_path = Path(payload.pop("h5ad_path"))
    return h5ad_path, payload


def load_adata_from_checkpoint(ckpt_path: Path) -> anndata.AnnData:
    h5ad_path, _ = load_checkpoint(ckpt_path)
    log.info("Resuming from checkpoint %s → loading %s", ckpt_path.name, h5ad_path)
    return anndata.read_h5ad(h5ad_path)


def md5_file(path: Path, chunk_size: int = 65_536) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while chunk := fh.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()
