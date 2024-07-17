from contextlib import nullcontext
import torch
import torch.nn as nn
from transformers import CLIPTextModelWithProjection

from modules.build import LANGUAGE_REGISTRY
from modules.utils import get_mlp_head, layer_repeat
from modules.grounding.query_encoder import SelfAttentionLayer


@LANGUAGE_REGISTRY.register()
class CLIPLanguageEncoder(nn.Module):
    def __init__(self, cfg, weights="openai/clip-vit-large-patch14", output_dim=768, freeze_backbone=True, use_projection=False, projection_type='mlp', num_projection_layers=1, dropout=0.1):
        super().__init__()
        self.context = torch.no_grad if freeze_backbone else nullcontext
        self.model = CLIPTextModelWithProjection.from_pretrained(weights)
        self.use_projection = use_projection
        self.projection_type = projection_type
        if use_projection:
            if projection_type == 'mlp':
                self.projection = get_mlp_head(self.model.config.hidden_size, output_dim, output_dim, dropout=dropout)
            elif projection_type == 'attention':
                self.projection = layer_repeat(SelfAttentionLayer(self.model.config.hidden_size, nhead=12, dropout=dropout, normalize_before=False, batch_first=True), num_projection_layers)
            else:
                raise NotImplementedError
        #self.attention = nn.MultiheadAttention(embed_dim=768, num_heads=12, batch_first=True)
        
    def forward(self, txt_ids, txt_masks):
        with self.context():
            txt = self.model(txt_ids, txt_masks).last_hidden_state
            txt = self.model.text_projection(txt)
            txt = torch.nn.functional.normalize(txt, p=2, dim=2)
        #txt = self.attention(txt, txt, txt, key_padding_mask=txt_masks.logical_not())[0]
        if self.use_projection:
            if self.projection_type == 'mlp':
                txt = self.projection(txt)
            elif self.projection_type == 'attention':
                for attention_layer in self.projection:
                    txt = attention_layer(txt, tgt_key_padding_mask = txt_masks.logical_not())
            else:
                raise NotImplementedError
        return txt