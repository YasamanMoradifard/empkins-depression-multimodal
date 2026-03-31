"""
-*- coding: utf-8 -*-
@Author     :   Md Rezwanul Haque
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseNet
from .Generate_Audio_Model import AudioTransformerModel
from .Generate_Visual_Model import GenerateVisualModel
from .transformer_timm import AttentionBlock, Attention
from .mutualtransformer import MutualTransformer

class MultiModalDepDet(BaseNet):

    def __init__(self, audio_input_size=161, video_input_size=161, mm_input_size=128, 
                 mm_output_sizes=[128, 256, 768], 
                 fusion='ia', num_heads=4,
                 # ========== NEW: Dropout parameters (Added for regularization control) ==========
                 fusion_dropout=0.5, audio_dropout=0.5, visual_dropout=0.5,
                 attention_dropout=0.1, transformer_dropout=0.1, classifier_dropout=0.0):
                 # ========== END NEW: Dropout parameters ==========
        super().__init__()

        self.fusion  =   fusion

        self.conv_audio = nn.Conv1d(audio_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        self.conv_video = nn.Conv1d(video_input_size, mm_input_size, 1, padding=0, dilation=1, bias=False)
        
        self.audio_model = AudioTransformerModel(input_tdim=audio_input_size, 
                                                    label_dim=2, audioset_pretrain=True)
        
        self.visual_model = GenerateVisualModel(temporal_layers=6, number_class=2)

        # Initialize mutual transformer for crossing and fusing: video & audio
        # ========== NEW: Pass transformer_dropout to MutualTransformer (Added for regularization) ==========
        self.mutualtransformer = MutualTransformer(a_d=145, v_d=128, transformer_dropout=transformer_dropout)
        # ========== END NEW ==========

        # Encoder for fused audio-visual features
        # ========== NEW: Use transformer_dropout parameter (Added for regularization consistency) ==========
        self.av_encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=3*256, nhead=8, dim_feedforward=6*256, 
                dropout=transformer_dropout,  # NEW: Use transformer_dropout parameter
                batch_first=True
            ),
            num_layers=6,
        )
        # ========== END NEW ==========

        ## audio conv
        self.conv1d_block_audio  = self.conv1d_block(in_channels=768, out_channels=512)
        self.conv1d_block_audio_1 = self.conv1d_block(512, 256)
        self.conv1d_block_audio_2 = self.conv1d_block(256, 128)
        self.conv1d_block_audio_3 = self.conv1d_block(128, 128)

        ## video conv
        self.conv1d_block_visual = self.conv1d_block(in_channels=768, out_channels=512)
        self.conv1d_block_visual_1 = self.conv1d_block(512, 256)
        self.conv1d_block_visual_2 = self.conv1d_block(256, 128)
        self.conv1d_block_visual_3 = self.conv1d_block(128, 128)
        
        self.pool = nn.AdaptiveMaxPool1d(1)

        ## [128, 129, 256, 273, 768]
        if self.fusion in ['audio', 'video', 'MT']:
            self.output = nn.Linear(mm_output_sizes[4], 1) ##  768
        elif self.fusion in ['add', 'multi']:
            self.output = nn.Linear(mm_output_sizes[0], 1) ##  128
        elif self.fusion in ['concat']:
            self.output = nn.Linear(mm_output_sizes[3], 1) ##  273
        elif self.fusion in ['tensor_fusion']:
            self.output = nn.Linear(mm_output_sizes[1], 1) ##  129
        else:
            self.output = nn.Linear(mm_output_sizes[2], 1) ##  256

        self.m = nn.Sigmoid()

        nn.init.xavier_uniform_(self.conv_audio.weight.data)
        nn.init.xavier_uniform_(self.conv_video.weight.data)

        ## 
        e_dim               =   128
        input_dim_video     =   128
        input_dim_audio     =   128

        if self.fusion in ['lt', 'it']:
            if self.fusion  == 'lt':
                # ========== NEW: Use attention_dropout parameter (Added for regularization) ==========
                # Note: AttentionBlock uses 'drop' for proj_drop, not 'proj_drop'
                self.av = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=e_dim, num_heads=num_heads, attn_drop=attention_dropout, drop=attention_dropout)
                self.va = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=e_dim, num_heads=num_heads, attn_drop=attention_dropout, drop=attention_dropout)
                # ========== END NEW ==========
            elif self.fusion == 'it':
                # ========== NEW: Use attention_dropout parameter (Added for regularization) ==========
                # Note: AttentionBlock uses 'drop' for proj_drop, not 'proj_drop'
                self.av1 = AttentionBlock(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads, attn_drop=attention_dropout, drop=attention_dropout)
                self.va1 = AttentionBlock(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads, attn_drop=attention_dropout, drop=attention_dropout)
                # ========== END NEW ==========
        
        elif self.fusion in ['ia']:
            # ========== NEW: Use attention_dropout parameter (Added for regularization) ==========
            self.av1 = Attention(in_dim_k=input_dim_video, in_dim_q=input_dim_audio, out_dim=input_dim_audio, num_heads=num_heads, attn_drop=attention_dropout, proj_drop=attention_dropout)
            self.va1 = Attention(in_dim_k=input_dim_audio, in_dim_q=input_dim_video, out_dim=input_dim_video, num_heads=num_heads, attn_drop=attention_dropout, proj_drop=attention_dropout)
            # ========== END NEW ==========

        ## dropout
        # ========== NEW: Use dropout parameters from arguments (Changed from hardcoded 0.5) ==========
        self.fusion_dropout = nn.Dropout(p=fusion_dropout)
        self.audio_dropout  = nn.Dropout(p=audio_dropout)
        self.visual_dropout = nn.Dropout(p=visual_dropout)
        self.classifier_dropout = nn.Dropout(p=classifier_dropout)  # NEW: Added classifier dropout
        # ========== END NEW ==========

    @staticmethod
    def conv1d_block(in_channels, out_channels, kernel_size=3, stride=1, padding='same'):
        """
        Creates a 1D convolutional block with optional padding.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Size of the convolutional kernel.
            stride (int): Stride of the convolution.
            padding (str): Padding method ('same' or 'valid').

        Returns:
            nn.Sequential: A sequential container of layers.
        """
        if padding == 'same':
            pad = kernel_size // 2  # Calculate padding for 'same' padding
        elif padding == 'valid':
            pad = 0
        else:
            raise ValueError("Padding must be 'same' or 'valid'")
        
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=pad),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=1)
        )
    
    def forward_audio_conv1d(self, x):
        x = self.conv1d_block_audio_1(x)
        x = self.conv1d_block_audio_2(x)
        return x 
    
    def forward_visual_conv1d(self, x):
        x = self.conv1d_block_visual_1(x)
        x = self.conv1d_block_visual_2(x)
        return x 
    
    def late_transformer_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])
       
        ### Late Transformer
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        h_av = self.av(proj_x_v, proj_x_a) # torch.Size([8, 143, 128])
        h_va = self.va(proj_x_a, proj_x_v) # torch.Size([8, 126, 128])

        return h_av, h_va
    
    def intermediate_transformer_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])
        
        ### Intermidiate Transformer
        proj_x_a = xa.permute(0, 2, 1)
        proj_x_v = xv.permute(0, 2, 1)
        h_av = self.av1(proj_x_v, proj_x_a) # torch.Size([8, 143, 128])
        h_va = self.va1(proj_x_a, proj_x_v) # torch.Size([8, 126, 128])

        h_av = h_av.permute(0,2,1)
        h_va = h_va.permute(0,2,1)

        xa = h_av + xa  # torch.Size([8, 128, 143])
        xv = h_va + xv  # torch.Size([8, 128, 126])

        xa = self.conv1d_block_audio_3(xa) # torch.Size([8, 128, 142])
        xv = self.conv1d_block_visual_3(xv) # torch.Size([8, 128, 125])

        h_av = xa.permute(0,2,1) # torch.Size([8, 142, 128])
        h_va = xv.permute(0,2,1) # torch.Size([8, 125, 128])

        return h_av, h_va

    def intermediate_attention_fusion(self, xa, xv):
        xa = self.forward_audio_conv1d(xa)
        proj_x_a = xa ## torch.Size([8, 128, 143])

        xv = self.forward_visual_conv1d(xv)
        proj_x_v = xv # torch.Size([8, 128, 126])

        ### Intermidiate Attention 
        proj_x_a = xa.permute(0, 2, 1)
        proj_x_v = xv.permute(0, 2, 1)
        _, h_av = self.av1(proj_x_v, proj_x_a) ## torch.Size([8, 8, 143, 126])
        _, h_va = self.va1(proj_x_a, proj_x_v) ## torch.Size([8, 8, 126, 143])

        if h_av.size(1) > 1: #if more than 1 head, take average
            h_av = torch.mean(h_av, axis=1).unsqueeze(1)
        h_av = h_av.sum([-2]) ## torch.Size([8, 1, 126]

        if h_va.size(1) > 1: #if more than 1 head, take average
            h_va = torch.mean(h_va, axis=1).unsqueeze(1)
        h_va = h_va.sum([-2]) ## torch.Size([8, 1, 143])
 
        xa = h_va * xa  # torch.Size([8, 128, 143])
        xv = h_av * xv  # torch.Size([8, 128, 126])

        xa = self.conv1d_block_audio_3(xa) # torch.Size([8, 128, 142])
        xv = self.conv1d_block_visual_3(xv) # torch.Size([8, 128, 125])

        h_av = xa.permute(0,2,1) # torch.Size([8, 142, 128])
        h_va = xv.permute(0,2,1) # torch.Size([8, 125, 128])

        return h_av, h_va


    def feature_extractor(self, x, padding_mask=None):
        # Get expected dimensions from conv layers
        expected_audio_size = self.conv_audio.weight.shape[1]  # input channels
        expected_video_size = self.conv_video.weight.shape[1]  # input channels
        
        # Determine mode based on fusion type and input dimensions
        if self.fusion == 'audio':
            # Audio-only mode: all features are audio
            xa = x
            batch_size, seq_len = x.shape[0], x.shape[1]
            xv = torch.zeros(batch_size, seq_len, expected_video_size, 
                           device=x.device, dtype=x.dtype)  # Zero video features
        elif self.fusion == 'video':
            # Video-only mode: all features are video
            xv = x
            batch_size, seq_len = x.shape[0], x.shape[1]
            xa = torch.zeros(batch_size, seq_len, expected_audio_size, 
                           device=x.device, dtype=x.dtype)  # Zero audio features
        elif x.shape[2] > expected_video_size:
            # Audio+video mode: split at video size
            xa = x[:, :, expected_video_size:] ## xa: torch.Size([8, ~3404, audio_input_size])
            xv = x[:, :, :expected_video_size] ## xv: torch.Size([8, ~3404, 136])
        else:
            # Fallback: assume video-only (for backward compatibility)
            # This handles cases where fusion is not 'audio' or 'video' but input is <= video size
            xv = x
            batch_size, seq_len = x.shape[0], x.shape[1]
            xa = torch.zeros(batch_size, seq_len, expected_audio_size, 
                           device=x.device, dtype=x.dtype)  # Zero audio features
        
        xa = self.conv_audio(xa.permute(0,2,1)).permute(0,2,1) ## torch.Size([8, ~1570, 256])
        xv = self.conv_video(xv.permute(0,2,1)).permute(0,2,1) ## torch.Size([8, ~1570, 256])
       
        xa = self.audio_model.forward(xa) # torch.Size([8, 146, 768])
        xv = self.visual_model.forward_visual(xv) # torch.Size([8, 129, 768])

        ## dropout
        xa = self.audio_dropout(xa)
        audio_feat = xa 
        xv = self.visual_dropout(xv)
        video_feat = xv 

        ## `Conv1D block`
        xa = xa.permute(0, 2, 1)  # Shape: [8, 768, 146]
        xa = self.conv1d_block_audio(xa) # torch.Size([8, 512, 145]) 
        
        xv = xv.permute(0, 2, 1)  # Shape: [8, 768, 129]
        xv = self.conv1d_block_visual(xv)  # torch.Size([8, 512, 128])

        ##-------------------------------------------------------------------
        ## Single Modality ==> [audio, video]
        if self.fusion == 'audio':
            # print(f"audio_feat: {audio_feat.shape}")
            z = audio_feat.mean([1]) # torch.Size([8, 768]) # mean accross temporal dimension
            z = self.fusion_dropout(z)
            # print(f"z: {z.shape}")
            return z

        elif self.fusion == 'video':
            # print(f"video_feat: {video_feat.shape}")
            z = video_feat.mean([1]) # torch.Size([8, 768]) # mean accross temporal dimension
            z = self.fusion_dropout(z)
            # print(f"z: {z.shape}")
            return z

        ##-------------------------------------------------------------------
        ## Proposed fusion strategies ==> [LT, IT, IA, MT]
        elif self.fusion == 'lt':
            h_av, h_va = self.late_transformer_fusion(xa, xv)

            audio_pooled = h_av.mean([1]) # torch.Size([8, 128]) # mean accross temporal dimension
            video_pooled = h_va.mean([1]) # torch.Size([8, 128])

            x = torch.cat((audio_pooled, video_pooled), dim=-1)  # torch.Size([8, 256])

            x = self.fusion_dropout(x)
            return x

        elif self.fusion == 'it':
            h_av, h_va = self.intermediate_transformer_fusion(xa, xv)

            audio_pooled = h_av.mean([1]) # torch.Size([8, 128]) # mean accross temporal dimension
            video_pooled = h_va.mean([1]) # torch.Size([8, 128])

            x = torch.cat((audio_pooled, video_pooled), dim=-1)  # torch.Size([8, 256])

            x = self.fusion_dropout(x)
            return x

        elif self.fusion == 'ia':
            h_av, h_va = self.intermediate_attention_fusion(xa, xv)

            audio_pooled = h_av.mean([1]) # torch.Size([8, 128]) # mean accross temporal dimension
            video_pooled = h_va.mean([1]) # torch.Size([8, 128])

            x = torch.cat((audio_pooled, video_pooled), dim=-1)  # torch.Size([8, 256])

            x = self.fusion_dropout(x)
            return x

        elif self.fusion == 'MT':
            mutual_fused_feats = self.mutualtransformer(xa, xv) # torch.Size([8, 512, 768])

            #1
            # pooled_adaptive_avg = F.adaptive_avg_pool1d(mutual_fused_feats.transpose(1, 2), 1).squeeze(-1)
            # pooled_adaptive_max = F.adaptive_max_pool1d(mutual_fused_feats.transpose(1, 2), 1).squeeze(-1)
            # pooled_features = torch.cat([pooled_adaptive_avg, pooled_adaptive_max], dim=1) # concatenate along the feature dimension. #torch.Size([8, 1536])
            # z = pooled_features
            
            #2
            # ## Encoder for fused audio-visual features
            # xav_fused = self.av_encoder(mutual_fused_feats)
            # z = torch.mean(xav_fused, dim=1)
           
            # #3
            # mean accross temporal dimension
            z = torch.mean(mutual_fused_feats, dim=1) # torch.Size([8, 768])

            z = self.fusion_dropout(z)
            return z
        ##-------------------------------------------------------------------
        ### Abalation Study ==> [add, multi, concat, tensor_fusion]
        elif self.fusion =='add':
            # print(xa.shape, xv.shape) ## torch.Size([8, 512, 145]) torch.Size([8, 512, 128])

            ## Element wise adddistion #  Truncation:
            min_len = min(xa.shape[-1], xv.shape[-1])
            x_add = xa[:, :, :min_len] + xv[:, :, :min_len]
            # print(f"x_ablation shape after truncation: {x_add.shape}") # Output: torch.Size([8, 512, 128])

            x_add_h_pool = x_add.mean([1]) # mean accross temporal dimension
            # print(f"x_add_h_pool: {x_add_h_pool.shape}") # x_add_h_pool: torch.Size([8, 128])

            z = self.fusion_dropout(x_add_h_pool)
            return z

        elif self.fusion == 'multi':
            # print(xa.shape, xv.shape) ## torch.Size([8, 512, 145]) torch.Size([8, 512, 128])

            ## Element wise multiplication with Truncation:
            min_len = min(xa.shape[-1], xv.shape[-1])
            x_mul = xa[:, :, :min_len] * xv[:, :, :min_len]
            # print(f"x_ablation shape after truncation: {x_mul.shape}") # Output: torch.Size([8, 512, 128])

            x_mul_h_pool = x_mul.mean([1]) # mean across temporal dimension
            # print(f"x_mul_h_pool: {x_mul_h_pool.shape}") # x_mul_h_pool: torch.Size([8, 128])

            z = self.fusion_dropout(x_mul_h_pool)
            return z

        elif self.fusion == 'concat':
            ## Concatenation along the last dimension
            x_concat = torch.cat((xa, xv), dim=-1)
            # print(f"x concat: {x_concat.shape}") ## torch.Size([8, 512, 273])

            x_concat_h_pool = x_concat.mean([1]) # mean across temporal dimension
            # print(f"x_concat_h_pool: {x_concat_h_pool.shape}") # torch.Size([8, 273])

            z = self.fusion_dropout(x_concat_h_pool)
            return z
        
        elif self.fusion == 'tensor_fusion':
            ## Tensor Fusion Network (TFN) inspired fusion
            batch_size, num_modalities, feature_dim_xa = xa.shape[0], 2, xa.shape[-1]
            feature_dim_xv = xv.shape[-1]
            min_feature_dim = min(feature_dim_xa, feature_dim_xv)

            # Truncate features
            xa_truncated = xa[:, :, :min_feature_dim] # [batch_size, 512, min_feature_dim]
            xv_truncated = xv[:, :, :min_feature_dim] # [batch_size, 512, min_feature_dim]

            # Polynomial expansion (simplified for two modalities)
            xa_poly = torch.cat([torch.ones(batch_size, 512, 1).to(xa.device), xa_truncated], dim=-1) # [batch_size, 512, min_feature_dim + 1]
            xv_poly = torch.cat([torch.ones(batch_size, 512, 1).to(xv.device), xv_truncated], dim=-1) # [batch_size, 512, min_feature_dim + 1]

            # Outer product for fusion (across the feature dimension)
            fused_tensor = torch.einsum('bij,bik->bijk', xa_poly, xv_poly) # [batch_size, 512, min_feature_dim + 1, min_feature_dim + 1]

            # Mean across the temporal dimension
            fused_pooled = fused_tensor.mean(dim=1) # [batch_size, min_feature_dim + 1, min_feature_dim + 1]
            # print(fused_pooled.shape) # torch.Size([8, 129, 129])

            # Flatten for dropout
            fused_flattened = fused_pooled.mean([1]) # mean across temporal dimension
            # print(fused_flattened.shape) # torch.Size([8, 129])

            z = self.fusion_dropout(fused_flattened)
            return z
        ##-------------------------------------------------------------------

    def classifier(self, x):
        # ========== NEW: Add dropout before final classifier (Added for regularization) ==========
        x = self.classifier_dropout(x)  # Apply dropout before final output layer
        # ========== END NEW ==========
        return self.output(x)
