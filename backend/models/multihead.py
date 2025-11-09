import json
import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class _MultiHeadDistilBert(nn.Module):
    def __init__(self, base_name: str, num_labels_sent: int, num_labels_emot: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_name)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.classifier_sent = nn.Linear(h, num_labels_sent)
        self.classifier_emot = nn.Linear(h, num_labels_emot)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]
        x = self.dropout(cls)
        return self.classifier_sent(x), self.classifier_emot(x)


class MultiheadModel:
    def __init__(self, export_dir: str, device: str | None = None, max_len: int = 256):
        self.name = "multihead"
        self.export_dir = export_dir
        self.max_len = max_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        with open(os.path.join(export_dir, "meta.json"), "r") as f:
            meta = json.load(f)

        self.id2label_sent = meta.get(
            "id2label_sent", {str(i): str(i) for i in range(meta["num_labels_sent"])}
        )
        self.id2label_emot = meta.get(
            "id2label_emot", {str(i): str(i) for i in range(meta["num_labels_emot"])}
        )

        self.tok = AutoTokenizer.from_pretrained(export_dir)
        self.model = _MultiHeadDistilBert(
            base_name=meta["base_name"],
            num_labels_sent=meta["num_labels_sent"],
            num_labels_emot=meta["num_labels_emot"],
        )
        state = torch.load(
            os.path.join(export_dir, "pytorch_model.bin"), map_location="cpu"
        )
        self.model.load_state_dict(state, strict=True)
        self.model.to(self.device).eval()

    def predict(self, texts: List[str]) -> List[Dict[str, Any]]:
        enc = self.tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            ls, le = self.model(enc["input_ids"], enc["attention_mask"])
            ps, pe = F.softmax(ls, dim=-1), F.softmax(le, dim=-1)

        outs = []
        for i in range(len(texts)):
            s_idx = int(ps[i].argmax().item())
            e_idx = int(pe[i].argmax().item())
            outs.append(
                {
                    "sentiment": self.id2label_sent[str(s_idx)],
                    "sentiment_probs": {
                        self.id2label_sent[str(j)]: float(ps[i, j])
                        for j in range(ps.shape[1])
                    },
                    "emotion": self.id2label_emot[str(e_idx)],
                    "emotion_probs": {
                        self.id2label_emot[str(j)]: float(pe[i, j])
                        for j in range(pe.shape[1])
                    },
                }
            )
        return outs
