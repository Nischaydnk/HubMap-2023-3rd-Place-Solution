import argparse

import timm
from timm.models.vision_transformer import VisionTransformer
from timm.models.vision_transformer import *
import torch
import torch.nn as nn

import torch
import torch.nn as nn
from timm.models import create_model
from functools import partial
from PIL import Image

import torch.nn as nn
from ..builder import BACKBONES
from torchvision.transforms import Resize
from einops import rearrange
from segmentation_models_pytorch.encoders import get_encoder
import torch.nn.functional as F

import os 

class Transformer_Encoder(VisionTransformer):
    def __init__(self, pretrained = False, pretrained_model = None, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=768, depth=12,
                  num_heads=12, mlp_ratio=4., qkv_bias=False, drop_rate=0., attn_drop_rate=0.,
                  drop_path_rate=0., norm_layer=nn.LayerNorm):

        super(Transformer_Encoder, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=1000, embed_dim=embed_dim, depth=depth,
                  num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
                  drop_path_rate=drop_path_rate, norm_layer=norm_layer)
        
        self.num_classes = 1
        self.dispatcher = {
            'vit_small_patch16_224': vit_small_patch16_224,
            'vit_base_patch16_224': vit_base_patch16_224,
            'vit_large_patch16_224': vit_large_patch16_224,
            'vit_base_patch16_384': vit_base_patch16_384,
            'vit_base_patch32_384': vit_base_patch32_384,
            'vit_large_patch16_384': vit_large_patch16_384,
            'vit_large_patch32_384': vit_large_patch32_384,
            'vit_large_patch16_224' : vit_large_patch16_224,
            'vit_large_patch32_384': vit_large_patch32_384,
            # 'vit_small_resnet26d_224': vit_small_resnet26d_224,
            # 'vit_small_resnet50d_s3_224': vit_small_resnet50d_s3_224,
            # 'vit_base_resnet26d_224' : vit_base_resnet26d_224,
            # 'vit_base_resnet50d_224' : vit_base_resnet50d_224,
        }
        self.pretrained_model = pretrained_model
        self.pretrained = pretrained
        if pretrained:
            self.load_weights()
        self.head = nn.Identity()
        self.encoder_out = [1,2,3,4,5]

        

    def forward_features(self, x):

        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        features = []

        for i,blk in enumerate(self.blocks,1):
            x = blk(x)
            if i in self.encoder_out:
                features.append(x)

        for i in range(len(features)):
            features[i] = self.norm(features[i])

        return features

    def forward(self, x):

        features = self.forward_features(x)
        return features
    
    def load_weights(self):
        model = None
        try:
            model = self.dispatcher[self.pretrained_model](pretrained=True)
        except:
            print('could not not load model')
        if model == None:
            return
        # try:
        self.load_state_dict(model.state_dict())
        print("successfully loaded weights!!!")
        
        # except:
        #     print("Could not load weights. Parameters should match!!!")

@BACKBONES.register_module()
class Viteff(nn.Module):
    def __init__(self, **kwargs):

        super().__init__()
        self.num_classes = 1
        self.emb_dim = 768
        self.pretrained = True
        self.pretrained_trans_model = 'vit_base_patch16_384'
        self.patch_size = 16

        self.encoder_name = 'timm-efficientnet-b5'
        self.in_channels = 3
        self.encoder_depth = 5
        self.encoder_weights = 'noisy-student'
        
        self.conv_encoder = get_encoder(self.encoder_name,
                in_channels=self.in_channels,
                depth=self.encoder_depth,
                weights=self.encoder_weights)
        self.conv_encoder.num_classes = 1
        
        self.conv_channels = self.conv_encoder.out_channels
        
        self.transformer = Transformer_Encoder(pretrained = True, img_size = 384, pretrained_model = self.pretrained_trans_model, patch_size=16, embed_dim=768, depth=12, num_heads=12, qkv_bias = True)
        self.conv_final = nn.ModuleList(
              [nn.Conv2d(self.conv_channels[i],self.emb_dim,3,stride = 2, padding = 1) for i in range(1,len(self.conv_channels))]
          )
        self.names = ["p"+str(i+2) for i in range(5)]
        self.resize =  Resize((384,384))
        self.Wq = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        self.Wk = nn.Linear(self.emb_dim, self.emb_dim, bias = False)
        self.output_conv = nn.ModuleList([nn.Conv2d(self.emb_dim,2**(8+i),1,stride=1,padding = 0) for i in range(4)])
            
        # else :
          # self.output_conv = nn.ModuleList([nn.Conv2d(self.conv_channels[i+2],2**(8+i),1,stride=1,padding = 0) for i in range(4)])
        

    def forward(self, image):
    
        conv_features = list(self.conv_encoder(image))
    
        conv_features = conv_features[1:]
        for i in range(len(self.conv_final)):
            conv_features[i]  = self.conv_final[i](conv_features[i])        
        exp_shape = [i.shape for i in conv_features]
        transformer_features = self.transformer(F.interpolate(image,(384,384)))
        features = self.project(conv_features, transformer_features)
        features = self.emb2img(features, exp_shape)
        features = features[:-1]
        features = [self.output_conv[i](features[i]) for i in range(len(features))]
        features.insert(0,image)

      # else :
      #   conv_features = conv_features[2:]
      #   features = conv_features
      #   features = [self.output_conv[i](features[i]) for i in range(len(features))]
      #   features.insert(0,image)

        return features
    
    def project(self, conv_features, transformer_features):

        features = []

        for i in range(len(conv_features)):

            t = transformer_features[i]
            x = rearrange(conv_features[i], 'b c h w -> b (h w) c') 
            xwq = self.Wq(x)
            twk = self.Wk(t)
            twk_T = rearrange(twk, 'b l c -> b c l')
            A = torch.einsum('bij,bjk->bik', xwq, twk_T).softmax(dim = -1)
            x += torch.einsum('bij,bjk->bik', A, t)
            features.append(x)

        return features
    
    def emb2img(self, features, exp_shape):

        for i, x in enumerate(features):
            B, P, E = x.shape             #(batch_size, latent_dim, emb_dim)
            x = x.transpose(1,2).reshape(B, E, exp_shape[i][2], exp_shape[i][3])
            features[i] = x

        return features
    
    def init_weights(self, pretrained=None):
      pass
    


