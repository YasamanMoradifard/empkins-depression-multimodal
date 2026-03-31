'''
-*- coding: utf-8 -*-
@Author     :   Md Rezwanul Haque
@Adapted    :   https://github.com/YuanGongND/ast/blob/master/src/models/ast_models.py
@Paper      :   https://arxiv.org/pdf/2104.01778
@Description:   This is for Audio Feature Extraction!
'''
import torch
import torch.nn as nn
from torch.amp import autocast 
import os
import wget
from pathlib import Path
# Set TORCH_HOME to absolute path
pretrained_models_dir = Path(__file__).parent.parent / "pretrained_models"
pretrained_models_dir.mkdir(parents=True, exist_ok=True)
os.environ['TORCH_HOME'] = str(pretrained_models_dir)
import timm
from timm.models.layers import to_2tuple, trunc_normal_

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class AudioTransformerModel(nn.Module):
    def __init__(self, label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=True, audioset_pretrain=False, model_size='base384', verbose=False):
        super(AudioTransformerModel, self).__init__()
        assert timm.__version__ == '0.4.5', 'Please use timm == 0.4.5'
        timm.models.vision_transformer.PatchEmbed = PatchEmbed

        self.fstride = fstride
        self.tstride = tstride

        # Define downsampling layers: audio
        self.audio_downsample = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.BatchNorm1d(128),
            nn.AdaptiveAvgPool1d(128)  # Downsample to the desired length
        )

        if verbose:
            print('ImageNet pretraining: {:s}, AudioSet pretraining: {:s}'.format(str(imagenet_pretrain), str(audioset_pretrain)))

        if not audioset_pretrain:
            if model_size == 'tiny224':
                self.v = timm.create_model('vit_deit_tiny_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'small224':
                self.v = timm.create_model('vit_deit_small_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base224':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_224', pretrained=imagenet_pretrain)
            elif model_size == 'base384':
                self.v = timm.create_model('vit_deit_base_distilled_patch16_384', pretrained=imagenet_pretrain)
            else:
                raise ValueError('Model size must be one of tiny224, small224, base224, base384.')

            self.original_num_patches = self.v.patch_embed.num_patches
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))

            f_dim, t_dim = self.get_shape(fstride, tstride, input_fdim, input_tdim)
            self.v.patch_embed.num_patches = f_dim * t_dim
            new_proj = torch.nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
            if imagenet_pretrain:
                new_proj.weight = torch.nn.Parameter(torch.sum(self.v.patch_embed.proj.weight, dim=1).unsqueeze(1))
                new_proj.bias = self.v.patch_embed.proj.bias
            self.v.patch_embed.proj = new_proj

            if imagenet_pretrain:
                original_hw = int(self.original_num_patches ** 0.5)
                new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, self.original_num_patches, self.original_embedding_dim).transpose(1, 2).reshape(1, self.original_embedding_dim, original_hw, original_hw)
                self.register_buffer('base_pos_embed', new_pos_embed)
                self.v.pos_embed = nn.Parameter(torch.cat([self.v.pos_embed[:, :2, :].detach(), new_pos_embed.flatten(2).transpose(1, 2)], dim=1))
            else:
                self.base_pos_embed = None
                new_pos_embed = nn.Parameter(torch.zeros(1, self.v.patch_embed.num_patches + 2, self.original_embedding_dim))
                self.v.pos_embed = new_pos_embed
                trunc_normal_(self.v.pos_embed, std=.02)

        else:
            if not imagenet_pretrain:
                raise ValueError('AudioSet pretraining requires ImageNet pretraining.')
            if model_size != 'base384':
                raise ValueError('AudioSet pretraining only available for base384 model.')

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Get absolute path to pretrained_models directory
            model_file = Path(__file__).parent.parent / "pretrained_models" / "audioset_10_10_0.4593.pth"
            model_file.parent.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist
            
            if not model_file.exists():
                print(f"Downloading AudioSet pretrained model to {model_file}...")
                wget.download('https://www.dropbox.com/s/cv4knew8mvbrnvq/audioset_0.4593.pth?dl=1', out=str(model_file))
            sd = torch.load(str(model_file), map_location=device)
            audio_model = AudioTransformerModel(label_dim=527, fstride=10, tstride=10, input_fdim=128, input_tdim=1024, imagenet_pretrain=False, audioset_pretrain=False, model_size='base384', verbose=False)
            audio_model = torch.nn.DataParallel(audio_model)
            audio_model.load_state_dict(sd, strict=False)
            self.v = audio_model.module.v
            self.original_embedding_dim = self.v.pos_embed.shape[2]
            self.mlp_head = nn.Sequential(nn.LayerNorm(self.original_embedding_dim), nn.Linear(self.original_embedding_dim, label_dim))
            new_pos_embed = self.v.pos_embed[:, 2:, :].detach().reshape(1, 1212, 768).transpose(1, 2).reshape(1, 768, 12, 101)
            self.register_buffer('base_pos_embed', new_pos_embed)

    def get_shape(self, fstride, tstride, input_fdim=128, input_tdim=1024):
        test_input = torch.randn(1, 1, input_fdim, input_tdim)
        test_proj = nn.Conv2d(1, self.original_embedding_dim, kernel_size=(16, 16), stride=(fstride, tstride))
        test_out = test_proj(test_input)
        return test_out.shape[2], test_out.shape[3]

    @autocast('cuda')
    def forward(self, x):
        # print(f"1. x shape: {x.shape}")
        x = self.audio_downsample(x.transpose(1, 2))
        # print(f"1.1. x shape: {x.shape}")
        x = x.unsqueeze(1)
        # print(f"2. x shape: {x.shape}")
        x = x.transpose(2, 3)
        # print(f"3. x shape: {x.shape}")
        B, C, F, T = x.shape

        f_dim = (F - 16) // self.fstride + 1
        t_dim = (T - 16) // self.tstride + 1
        num_patches = f_dim * t_dim

        current_base_pos_embed = self.base_pos_embed
        new_pos_embed = torch.nn.functional.interpolate(current_base_pos_embed, size=(f_dim, t_dim), mode='bilinear')
        new_pos_embed = new_pos_embed.flatten(2).transpose(1, 2)

        cls_dist_pos_embed = self.v.pos_embed[:, :2, :].expand(B, -1, -1)
        total_pos_embed = torch.cat([cls_dist_pos_embed, new_pos_embed.expand(B, -1, -1)], dim=1)

        x = self.v.patch_embed(x)
        cls_tokens = self.v.cls_token.expand(B, -1, -1)
        dist_token = self.v.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)
        x = x + total_pos_embed
        x = self.v.pos_drop(x)

        for blk in self.v.blocks:
            x = blk(x)
        x = self.v.norm(x) ## torch.Size([16, 146, 768])

        # # print(f"x norm: {x.shape}")
        # x = (x[:, 0] + x[:, 1]) / 2 ## torch.Size([16, 768])
        # # print(f"x~~ norm: {x.shape}")
        # x = self.mlp_head(x) ## torch.Size([16, 2])
        # # print(f"x~~ mlp_head norm: {x.shape}")

        return x

if __name__ == '__main__':
    input_tdim = 1320
    ast_mdl = AudioTransformerModel(input_tdim=128, label_dim=10, audioset_pretrain=True)
    test_input = torch.rand([16, input_tdim, 256])
    test_output = ast_mdl(test_input)
    print(test_output.shape)


