from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from registry import ModelRegistry

app = FastAPI(title="Unified Inference API (local)")
REG = ModelRegistry("models.json")


class PredictIn(BaseModel):
    model: str = "multihead"  # choose which model to run
    text: Optional[str] = None
    texts: Optional[List[str]] = None


class PredictOut(BaseModel):
    model: str
    outputs: List[Dict[str, Any]]  # one dict per input text (model-specific shape)


@app.get("/healthz")
def health():
    return {"status": "ok", "models": REG.list_models()}


@app.get("/models")
def models():
    return {"available": REG.list_models()}


@app.post("/predict", response_model=PredictOut)
def predict(inp: PredictIn):
    texts = inp.texts if inp.texts else ([inp.text] if inp.text else [])
    if not texts:
        raise HTTPException(status_code=400, detail="Provide 'text' or 'texts'.")
    try:
        mdl = REG.get(inp.model)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    outputs = mdl.predict(texts)
    return {"model": inp.model, "outputs": outputs}
