from typing import Literal, Tuple

from torch import nn
from transformers import ViTModel, DeiTModel

pt_link = {
    'vit_l_16': 'google/vit-large-patch16-224-in21k',
    'vit_b_16': 'google/vit-base-patch16-224-in21k',
    'deit_small': 'facebook/deit-small-distilled-patch16-224',
    'deit_base': 'facebook/deit-base-distilled-patch16-224'
}

models = {
    'vit': ViTModel,
    'deit': DeiTModel
}


# Return CLS Token, Convolutional Projection, Transformer Encoder
def get_vision_backbone(
        arch: Literal[
            'vit_l_16',
            'vit_b_16',
            'deit_small',
            'deit_base'],
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    if arch not in pt_link:
        raise ValueError(f'Unknown architecture {arch}!')
    link = pt_link[arch]
    model = models[arch.split('_')[0]].from_pretrained(link)
    cls_token = model.embeddings.cls_token
    proj = model.embeddings.patch_embeddings.projection
    encoder = model.encoder.layer
    return cls_token, proj, encoder


print(get_vision_backbone('vit_l_16'))
