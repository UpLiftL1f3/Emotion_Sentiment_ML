import torch
import torch.nn as nn
from transformers import AutoModel


class MultiHeadDistilBert(nn.Module):
    """
    DistilBERT backbone with two classification heads:
      - sentiment head: num_labels_sent classes
      - emotion head:   num_labels_emot classes

    forward(**enc) returns:
      {"logits_sent": Tensor[B, S], "logits_emot": Tensor[B, E]}
    """

    def __init__(self, base_name: str, num_labels_sent: int, num_labels_emot: int):
        super().__init__()
        self.base_name = base_name
        self.backbone = AutoModel.from_pretrained(
            base_name
        )  # e.g., "distilbert-base-uncased"
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(0.2)
        self.sent_head = nn.Linear(hidden, num_labels_sent)
        self.emot_head = nn.Linear(hidden, num_labels_emot)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT: out.last_hidden_state [B, T, H]; use CLS token = index 0
        cls = out.last_hidden_state[:, 0, :]
        cls = self.dropout(cls)
        logits_sent = self.sent_head(cls)
        logits_emot = self.emot_head(cls)
        return {"logits_sent": logits_sent, "logits_emot": logits_emot}
