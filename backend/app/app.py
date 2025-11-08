# app/app.py
import json
import os

import torch
from app.models.multihead import MultiHeadDistilBert  # <- note the "app." prefix
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer


class Inp(BaseModel):
    text: str


def load_multihead_model(export_dir: str):
    with open(os.path.join(export_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    tok = AutoTokenizer.from_pretrained(export_dir)
    mdl = MultiHeadDistilBert(
        base_name=meta["base_name"],
        num_labels_sent=meta["num_labels_sent"],
        num_labels_emot=meta["num_labels_emot"],
    )
    sd = torch.load(os.path.join(export_dir, "pytorch_model.bin"), map_location="cpu")
    mdl.load_state_dict(sd, strict=True)
    mdl.eval()
    return tok, mdl, meta


EXPORTS = {
    "v1": "exports/export_multihead",
    "v2": "exports/export_multihead_alt1",
    "v3": "exports/export_multihead_alt2",
}

app = FastAPI()

# Allow your Vite dev server
VITE_ORIGIN = os.getenv("VITE_ORIGIN", "http://localhost:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[VITE_ORIGIN],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _load_all_models_once():
    # store in app.state so workers share-at-process level
    MODELS = {}
    for name, path in EXPORTS.items():
        MODELS[name] = load_multihead_model(path)
    app.state.MODELS = MODELS
    # small perf niceties
    torch.set_grad_enabled(False)
    torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "1")))


@torch.inference_mode()
@app.post("/predict_compare")
def predict_compare(inp: Inp):
    text = inp.text
    results = {}
    for name, (tok, mdl, meta) in app.state.MODELS.items():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True)
        out = mdl(**enc)  # assumes dict with 'logits_sent' and 'logits_emot'
        logits_s = out["logits_sent"]
        logits_e = out["logits_emot"]

        probs_s = torch.softmax(logits_s, dim=-1)[0].tolist()
        probs_e = torch.softmax(logits_e, dim=-1)[0].tolist()

        top_s = int(torch.argmax(logits_s, dim=-1)[0])
        top_e = int(torch.argmax(logits_e, dim=-1)[0])
        label_s = meta.get("id2label_sent", {}).get(str(top_s), top_s)
        label_e = meta.get("id2label_emot", {}).get(str(top_e), top_e)

        results[name] = {
            "sentiment": {"label": label_s, "probs": probs_s},
            "emotion": {"label": label_e, "probs": probs_e},
        }

    def confidence(r):
        return max(max(r["sentiment"]["probs"]), max(r["emotion"]["probs"]))

    winner = max(results.items(), key=lambda kv: confidence(kv[1]))[0]
    return {"winner": winner, "models": results}
