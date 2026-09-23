"""Minimal REST service. Runs on a laptop; never logs document text."""
from __future__ import annotations

from typing import List, Optional

from fastapi import FastAPI
from pydantic import BaseModel, Field

from . import __version__, analyze_text, deidentify, available_policies

app = FastAPI(title="edshield", version=__version__)


class ExtractRequest(BaseModel):
    text: str = Field(..., max_length=200_000)
    model_name: Optional[str] = None
    labels: Optional[List[str]] = None


class DeidRequest(BaseModel):
    text: str = Field(..., max_length=200_000)
    policy: str = "ferpa"
    method: str = "mask"
    model_name: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__, "policies": available_policies()}


@app.post("/pii/extract")
def pii_extract(req: ExtractRequest):
    return analyze_text(req.text, model_name=req.model_name, labels=req.labels).to_dict()


@app.post("/pii/deidentify")
def pii_deidentify(req: DeidRequest):
    return deidentify(req.text, method=req.method, policy=req.policy, model_name=req.model_name).to_dict()
