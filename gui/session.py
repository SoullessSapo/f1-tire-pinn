"""Session state and background jobs, shared by every view.

Streamlit reruns the whole script on each interaction; what must survive a
rerun lives in `st.session_state`: the running job, the loaded dataset, the
last training result and a model reopened from disk.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st
from job import Job, Settings

ROOT = Path(__file__).resolve().parents[1]
ss = st.session_state


def init() -> None:
    for key in ("job", "data", "result", "loaded"):
        ss.setdefault(key, None)


def start(kind: str, settings: Settings) -> None:
    job = Job(kind, settings, cached_data=ss.data)
    ss.job = job
    job.start()


def harvest() -> None:
    """Move the output of a finished job into the session."""
    job: Job | None = ss.job
    if job is None or job.running or job.harvested:
        return
    job.harvested = True
    if job.data is not None:
        ss.data = (job.settings.data_key(), job.data)
    if job.result is not None:
        ss.result = job.result


def busy() -> bool:
    return ss.job is not None and ss.job.running


def available_models() -> list[Path]:
    base = ROOT / "outputs"
    return sorted({p.parent for p in base.rglob("pinn_weights.pt")}) if base.exists() else []
