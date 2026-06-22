from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

log = logging.getLogger(__name__)


class RRunner:
    """Locate Rscript once and call standalone R scripts via subprocess.

    Each R script lives in single_cell_curation/r_scripts/ and has its own
    argparse CLI (R argparse package). Python callers pass a dict of
    --flag: value pairs; RRunner builds the command and runs it.
    """

    def __init__(self) -> None:
        self._rscript: str | None = shutil.which("Rscript")
        if not self._rscript:
            raise RuntimeError(
                "Rscript not found on PATH. Install R >= 4.0 and ensure Rscript is in PATH."
            )
        self._scripts_dir = Path(__file__).parent.parent / "r_scripts"

    def run(
        self,
        script_name: str,
        args: dict[str, str],
        *,
        timeout: int = 3600,
    ) -> subprocess.CompletedProcess[str]:
        script_path = self._scripts_dir / script_name
        if not script_path.exists():
            raise FileNotFoundError(f"R script not found: {script_path}")
        cmd = [self._rscript, "--vanilla", str(script_path)]
        for flag, value in args.items():
            cmd += [flag, value]
        log.debug("Running R: %s", " ".join(cmd))
        result = subprocess.run(
            cmd, check=True, capture_output=True, text=True, timeout=timeout
        )
        if result.stderr:
            log.debug("Rscript stderr:\n%s", result.stderr.strip())
        return result

    @property
    def available(self) -> bool:
        return self._rscript is not None
