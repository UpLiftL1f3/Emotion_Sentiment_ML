# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import torch, json
from transformers import AutoTokenizer
from models.multihead import MultiHeadDistilBert  # your nn.Module class

class Inp(BaseModel):
    text: str

def load_multihead_model(export_dir: str):
    # load metadata + tokenizer
    with open(f"{export_dir}/meta.json", "r") as f:
        meta = json.load(f)
    tok = AutoTokenizer.from_pretrained(export_dir)
    # rebuild model and load weights
    mdl = MultiHeadDistilBert(
        base_name=meta["base_name"],
        num_labels_sent=meta["num_labels_sent"],
        num_labels_emot=meta["num_labels_emot"],
    )
    sd = torch.load(f"{export_dir}/pytorch_model.bin", map_location="cpu")
    mdl.load_state_dict(sd, strict=True)
    mdl.eval()
    return tok, mdl, meta

# Example: load several variants to compare per request
EXPORTS = {
    "v1": "export_multihead",            # from current notebook
    "v2": "export_multihead_alt1",       # another trained run
    "v3": "export_multihead_alt2",       # etc.
    # "v4": "export_multihead_alt3",
}
MODELS = {}
for name, path in EXPORTS.items():
    MODELS[name] = load_multihead_model(path)

app = FastAPI()

@torch.inference_mode()
@app.post("/predict_compare")
def predict_compare(inp: Inp):
    text = inp.text
    results = {}
    for name, (tok, mdl, meta) in MODELS.items():
        enc = tok(text, return_tensors="pt", truncation=True, padding=True)
        out = mdl(**enc)
        # assuming your forward returns dict with 'logits_sent' and 'logits_emot'
        logits_s = out["logits_sent"]
        logits_e = out["logits_emot"]
        probs_s = torch.softmax(logits_s, dim=-1)[0].tolist()
        probs_e = torch.softmax(logits_e, dim=-1)[0].tolist()

        # Top-1 labels (use meta label maps if present)
        top_s = int(torch.argmax(logits_s, dim=-1)[0])
        top_e = int(torch.argmax(logits_e, dim=-1)[0])
        label_s = meta.get("id2label_sent", {}).get(str(top_s), top_s)
        label_e = meta.get("id2label_emot", {}).get(str(top_e), top_e)

        results[name] = {
            "sentiment": {"label": label_s, "probs": probs_s},
            "emotion":   {"label": label_e, "probs": probs_e},
        }

    # pick "best" by highest max-prob across heads (customize as needed)
    def confidence(r):
        return max(max(r["sentiment"]["probs"]), max(r["emotion"]["probs"]))
    winner = max(results.items(), key=lambda kv: confidence(kv[1]))[0]

    return {"winner": winner, "models": results}
