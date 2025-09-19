import torch
import torch.nn as nn
from .efficientnet import EfficientNetB4
from .transformer import TransformerModel
from .prompt_learner import PromptLearner, PromptCrossAttention
import math
import torch.nn.functional as F

# Multi-attention module
class MultiAttentionModule(nn.Module):
    def __init__(self, in_channels, num_attentions=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, num_attentions, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_attentions)
        self.relu = nn.ReLU()
    def forward(self, x):
        attn_maps = self.relu(self.bn(self.conv(x)))
        return attn_maps

# Texture enhancement block
class TextureEnhanceBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(in_channels*2, in_channels, 3, padding=1)
        self.conv3 = nn.Conv2d(in_channels*3, in_channels, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        D = self.pool(x)
        texture = x - D
        out1 = self.relu(self.conv1(texture))
        out2 = self.relu(self.conv2(torch.cat([texture, out1], dim=1)))
        out3 = self.relu(self.conv3(torch.cat([texture, out1, out2], dim=1)))
        return out3

# Bilinear attention pooling
def bilinear_attention_pooling(feature_map, attention_maps):
    B, C, H, W = feature_map.size()
    N = attention_maps.size(1)
    feature_map = feature_map.unsqueeze(1)
    attention_maps = attention_maps.unsqueeze(2)
    weighted = feature_map * attention_maps
    bap = weighted.view(B, N, C, -1).sum(-1)
    bap_norm = F.normalize(bap, p=2, dim=-1)
    bap_norm = bap_norm.view(B, -1)
    return bap_norm

# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0).to(x.device)
        return x

# Token attention aggregator
class TokenAttentionAggregator(nn.Module):
    def __init__(self, in_dim, num_tokens=5):
        super().__init__()
        self.num_tokens = num_tokens
        self.attn_conv = nn.Conv2d(in_dim, num_tokens, kernel_size=1)
    def forward(self, feat):
        B, C, H, W = feat.shape
        attn_map = torch.softmax(self.attn_conv(feat).view(B, self.num_tokens, -1), dim=-1)
        feat_flat = feat.view(B, C, -1)
        tokens = torch.bmm(attn_map, feat_flat.transpose(1,2))
        return tokens

class MultiLabelModel(nn.Module):
    def __init__(self, num_classes, embed_size=256, num_heads=8, num_layers=2, num_attentions=4, num_tokens=5, use_prompt=True):
        super(MultiLabelModel, self).__init__()
        self.efficientnet = EfficientNetB4()
        self.texture_block = TextureEnhanceBlock(32)
        self.attn_module = MultiAttentionModule(160, num_attentions=num_attentions)
        self.token_aggregator = TokenAttentionAggregator(160, num_tokens=num_tokens)
        self.projector = nn.Linear(num_tokens*160, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size)
        self.transformer = TransformerModel(embed_size, num_heads, num_classes, num_layers)
        self.sigmoid = nn.Sigmoid()
        self.num_attentions = num_attentions
        self.num_tokens = num_tokens
        self.use_prompt = use_prompt
        
        # Input preprocessing convolution layer
        self.input_conv = nn.Conv2d(3, 48, kernel_size=3, padding=1, stride=2)
        
        # Prompt learning components
        if self.use_prompt:
            self.prompt_learner = PromptLearner(num_classes=num_classes, classnames=[str(i) for i in range(num_classes)])
            self.prompt_attn = PromptCrossAttention(embed_dim=embed_size, num_heads=num_heads)

    def forward(self, x, return_frame_predictions=False):
        b, t, c, h, w = x.shape
        # Input preprocessing
        x = self.input_conv(x.view(-1, c, h, w))
        l2_feat, l5_feat = self.efficientnet(x)
        # Texture enhancement
        texture_feat = self.texture_block(l2_feat)
        # Multi-attention
        attn_maps = self.attn_module(l5_feat)
        # Token attention aggregator
        tokens = self.token_aggregator(l5_feat)
        tokens = tokens.view(b, t, self.num_tokens, 160)
        tokens = tokens.reshape(b, t, -1)
        proj = self.projector(tokens)
        proj = self.pos_encoder(proj)
        
        # Prompt enhancement
        if self.use_prompt:
            prompt_tokens = self.prompt_learner()
            proj = self.prompt_attn(proj, prompt_tokens)
        
        if return_frame_predictions:
            # Frame-level prediction
            frame_predictions = []
            for i in range(t):
                frame_feat = proj[:, i:i+1, :]
                frame_out = self.transformer(frame_feat)
                frame_pred = self.sigmoid(frame_out)
                frame_predictions.append(frame_pred)
            frame_predictions = torch.stack(frame_predictions, dim=1)
            
            # Video-level prediction
            video_out = self.transformer(proj)
            video_pred = self.sigmoid(video_out)
            
            return video_pred, frame_predictions
        else:
            # Standard mode: video-level prediction only
            out = self.transformer(proj)
            return self.sigmoid(out)