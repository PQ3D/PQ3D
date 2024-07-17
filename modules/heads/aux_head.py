import torch.nn as nn

from modules.build import HEADS_REGISTRY
from modules.utils import get_mlp_head


@HEADS_REGISTRY.register()
class ClsHead(nn.Module):
    def __init__(self, cfg, input_size=768, hidden_size=768, cls_size=607, dropout=0.3):
        super().__init__()
        self.clf_head = get_mlp_head(
            input_size, hidden_size,
            cls_size, dropout=dropout
        )

    def forward(self, inputs, **kwargs):
        logits = self.clf_head(inputs)
        return logits
