import torch.nn as nn
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


from modules.build import HEADS_REGISTRY
@HEADS_REGISTRY.register()
class T5(nn.Module):
    def __init__(self, cfg, variant='t5-small', input_size=768, use_projection=True, **kwargs):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(variant)
        self.model.config.update(kwargs)
        hidden_size = self.model.config.d_model
        self.use_projection = use_projection
        if use_projection:
            self.input_proj = nn.Sequential(nn.Linear(input_size, hidden_size), nn.LayerNorm(hidden_size))
        else:
            assert input_size == hidden_size, "input_feat_size should be equal to hidden_size!"

    def forward(self, query_embeds, attention_masks, labels=None):
        if self.use_projection:
            query_embeds = self.input_proj(query_embeds)

        if labels is not None:
            outputs = self.model(encoder_outputs=[query_embeds], attention_mask=attention_masks, labels=labels)
            outputs = outputs.logits
        else:
            outputs = self.model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=query_embeds), attention_mask=attention_masks, do_sample=False)
            outputs = outputs[:, 1:] # remove the decoder start token for T5 generation output.
        return outputs