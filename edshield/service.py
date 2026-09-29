"""Minimal REST service. Runs on a laptop; never logs document text.

Requests can only name things that already exist on this machine: a built-in
policy, and "rules" or a model listed in models.jsonl. File paths and
arbitrary Hub ids are rejected, so a request cannot make the server read a
file or fetch a model.
"""
from __future__ import annotations

from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from . import __version__, ModelUnavailableError, analyze_text, deidentify, available_policies
from .ner import load_manifest
from .types import ALL_LABELS

app = FastAPI(title="edshield", version=__version__)


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=200_000)
    model_name: Optional[str] = None
    labels: Optional[List[str]] = None


class DeidRequest(BaseModel):
    text: str = Field(..., max_length=200_000)
    policy: str = "ferpa"
    method: Optional[Literal["mask", "replace", "hash", "shift_dates"]] = None
    model_name: Optional[str] = None


def _check_model(name: Optional[str]) -> None:
    allowed = ["rules"] + sorted(load_manifest())
    if name is not None and name not in allowed:
        raise HTTPException(status_code=400, detail=f"Unknown model '{name}'. Available: {allowed}")


def _check_policy(name: str) -> None:
    if name not in available_policies():
        raise HTTPException(status_code=400, detail=f"Unknown policy '{name}'. Available: {available_policies()}")


def _run(fn, *args, **kw):
    try:
        return fn(*args, **kw)
    except ModelUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except RuntimeError as exc:  # the verifier refused to return text with a leak in it
        raise HTTPException(status_code=422, detail="De-identification could not be verified; no text returned.") from exc


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "policies": available_policies()}


@app.post("/pii/extract")
def pii_extract(req: ExtractRequest):
    _check_model(req.model_name)
    unknown = sorted(set(req.labels or ()) - set(ALL_LABELS))
    if unknown:
        raise HTTPException(status_code=400, detail=f"Unknown labels {unknown}. Available: {ALL_LABELS}")
    return _run(analyze_text, req.text, model_name=req.model_name, labels=req.labels).to_dict()


@app.post("/pii/deidentify")
def pii_deidentify(req: DeidRequest):
    _check_model(req.model_name)
    _check_policy(req.policy)
    return _run(deidentify, req.text, method=req.method, policy=req.policy, model_name=req.model_name).to_dict()
