"""Raptor-inspired per-finding attention, with explicit missing-window masking.

Approach credit: dreaddevelopment, Apache2.0; see NOTICE.md. Reimplemented and
adapted to grouped six-slot cache; these weights are not Raptor-compatible.
"""
import torch
from torch import nn
from torch.nn import functional as F
import timm
from common import ARCH

class KneeModel(nn.Module):
    def __init__(self,arch=ARCH,res=224,pretrained=False,grad_checkpointing=False):
        super().__init__()
        self.backbone=timm.create_model(arch,pretrained=pretrained,num_classes=0,
                                        in_chans=3,global_pool='avg',img_size=res)
        if grad_checkpointing: self.backbone.set_grad_checkpointing(True)
        dim=self.backbone.num_features
        self.norm=nn.LayerNorm(dim)
        self.att=nn.Sequential(nn.Linear(dim,256),nn.Tanh(),nn.Dropout(.2),nn.Linear(256,12))
        self.class_weight=nn.Parameter(torch.empty(12,dim)); nn.init.trunc_normal_(self.class_weight,std=.02)
        self.class_bias=nn.Parameter(torch.zeros(12))
        # This timm CoAtNet uses0.5/0.5, NOT the common ResNet ImageNet stats.
        cfg=getattr(self.backbone,'pretrained_cfg',{})
        mean=cfg.get('mean',(.5,.5,.5)); std=cfg.get('std',(.5,.5,.5))
        if tuple(mean)!=(.5,.5,.5) or tuple(std)!=(.5,.5,.5):
            raise ValueError('Pinned CoAtNet pretrained normalization changed')
        self.register_buffer('mean',torch.tensor(mean).view(1,3,1,1))
        self.register_buffer('std',torch.tensor(std).view(1,3,1,1))
    def pool(self,features,mask):
        if not mask.any(dim=1).all(): raise ValueError('Cannot predict a study with no images')
        h=self.norm(features)
        logits=self.att(h).float().masked_fill(~mask[...,None],float('-inf'))
        attention=logits.softmax(dim=1).to(h.dtype)
        pooled=torch.einsum('bkn,bkf->bnf',attention,h)
        return (pooled*self.class_weight).sum(-1)+self.class_bias
    def forward(self,pixels,mask):
        # Normalization resides in the model, shared by training and inference.
        b,k=pixels.shape[:2]; flat=pixels.flatten(0,1)
        present=mask.flatten().bool()
        if not mask.any(dim=1).all(): raise ValueError('Empty study')
        x=flat[present].float()/255.
        feats=self.backbone((x-self.mean)/self.std)
        all_features=feats.new_zeros((b*k,feats.shape[-1]))
        all_features[present]=feats
        return self.pool(all_features.reshape(b,k,-1),mask.bool())

def masked_bce(logits,targets):
    valid=torch.isfinite(targets)
    if not valid.any(): raise ValueError('Batch has no labels')
    losses=F.binary_cross_entropy_with_logits(logits.float(),targets.nan_to_num(0),reduction='none')
    return losses[valid].mean()
