"""Where a training run reports to.

Two consumers, and they want different things. ``history.json`` is for the next
session reading the repository -- small, committed if it is worth committing,
readable without a network. Weights & Biases is for watching a run that is
still going, and for putting two runs beside each other.

Both get the same rows, and neither is required: a run with no wandb login
still logs to the file and still works. That matters more than it sounds --
losing a six-hour run to a missing API key would be an absurd way to lose it.

**Losses are not the measurement.** They fall because the network is fitting
whatever it was handed. This project has been fooled twice by numbers that
moved in the right direction and meant nothing, so the arena win rate is
logged as its own metric and everything else is labelled as diagnostics.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunLog:
    """A run's record. Writes a file always, wandb when it can."""

    directory: Path
    project: str = "pkcm-agent"
    name: str | None = None
    #: A stable id for this run, so a restart appends to the same wandb chart
    #: instead of starting a second one beside it. Derived from the output
    #: directory when nothing is passed: the directory is what ``--resume``
    #: keys on, so two runs that resume each other necessarily share it.
    run_id: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    #: Set false to skip wandb entirely, whatever the environment says.
    use_wandb: bool = True

    _run: Any = field(default=None, repr=False)
    _rows: list[dict] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        if not self.use_wandb:
            return
        self._run = _start_wandb(self.project, self.name, self.config,
                                 self.directory, self.run_id or _id_for(self.directory))

    @property
    def online(self) -> bool:
        return self._run is not None

    @property
    def url(self) -> str | None:
        return getattr(self._run, "url", None) if self._run is not None else None

    def log(self, row: dict[str, Any], step: int | None = None) -> None:
        self._rows.append(row)
        (self.directory / "history.json").write_text(
            json.dumps(self._rows, indent=2), encoding="utf-8")
        if self._run is not None:
            self._run.log(row, step=step)

    def summary(self, values: dict[str, Any]) -> None:
        """Final numbers -- the ones worth comparing across runs."""
        (self.directory / "summary.json").write_text(
            json.dumps(values, indent=2), encoding="utf-8")
        if self._run is not None:
            self._run.summary.update(values)

    def artifact(self, path: Path, kind: str = "model") -> None:
        if self._run is None or not path.exists():
            return
        try:
            import wandb

            item = wandb.Artifact(f"{self.project}-{kind}", type=kind)
            item.add_file(str(path))
            self._run.log_artifact(item)
        except Exception as error:  # pragma: no cover - never worth a crash
            print(f"  (wandb artifact skipped: {error})")

    def finish(self) -> None:
        if self._run is not None:
            self._run.finish()
            self._run = None


def _id_for(directory: Path) -> str:
    """A stable id from where the run writes.

    ``runs/fifth`` always means the same wandb run. That is the property that
    matters on a machine whose power goes off when its owner leaves the room:
    the loop resumes from ``state.pt`` and the chart resumes with it, rather
    than the dashboard filling up with fragments of one experiment.
    """
    import hashlib

    return hashlib.sha1(str(directory.resolve()).encode("utf-8")).hexdigest()[:16]


def _start_wandb(project: str, name: str | None, config: dict, directory: Path,
                 run_id: str | None = None):
    """Start a run, or return ``None`` and say why.

    Never raises. A training run that dies because a logger could not reach the
    network has confused what it is for.
    """
    try:
        import wandb
    except ImportError:
        print("  (wandb not installed -- logging to history.json only)")
        return None

    if not (os.environ.get("WANDB_API_KEY") or _has_stored_credentials()):
        print("  (wandb not logged in -- logging to history.json only)")
        print("   run 'wandb login' yourself if you want the dashboard; "
              "the key is yours and should not pass through here")
        return None

    try:
        return wandb.init(project=project, name=name, config=config,
                          dir=str(directory), reinit=True,
                          id=run_id, resume="allow")
    except Exception as error:  # pragma: no cover - offline, quota, anything
        print(f"  (wandb unavailable: {error} -- logging to history.json only)")
        return None


def _has_stored_credentials() -> bool:
    for candidate in (Path.home() / ".netrc", Path.home() / "_netrc"):
        try:
            if candidate.exists() and "api.wandb.ai" in candidate.read_text(
                    encoding="utf-8", errors="ignore"):
                return True
        except OSError:
            continue
    return bool((Path.home() / ".config" / "wandb" / "settings").exists())
