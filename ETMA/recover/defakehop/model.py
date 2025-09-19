#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefakeHop Model Definition

Deep learning-based DeepFake detection and restoration model architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class DefakeHopModel(nn.Module):
    """DefakeHop main model"""
    
    def __init__(self, input_channels: int = 3, output_channels: int = 3):
        """
        Initialize model
        
        Args:
            input_channels: Input channel number
            output_channels: Output channel number
        """
        super(DefakeHopModel, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Encoder
        self.encoder = Encoder(input_channels)
        
        # Decoder
        self.decoder = Decoder(output_channels)
        
        # Attention mechanism
        self.attention = SelfAttention(512)
        
        # Skip connections
        self.skip_connections = SkipConnections()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
            
        Returns:
            torch.Tensor: Restored image [B, C, H, W]
        """
        # Encode
        encoded_features = self.encoder(x)
        
        # Attention mechanism
        attended_features = self.attention(encoded_features)
        
        # Decode (with skip connections)
        restored_image = self.decoder(attended_features, self.skip_connections.get_features())
        
        return restored_image


class Encoder(nn.Module):
    """编码器模块"""
    
    def __init__(self, input_channels: int):
        super(Encoder, self).__init__()
        
        # 初始卷积层
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=1, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 下采样块
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 512)
        
        # 激活函数
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """编码过程"""
        # 初始特征
        x = self.relu(self.bn1(self.conv1(x)))
        
        # 下采样
        x1 = self.down1(x)    # 128
        x2 = self.down2(x1)   # 256
        x3 = self.down3(x2)   # 512
        x4 = self.down4(x3)   # 512
        
        return x4


class Decoder(nn.Module):
    """解码器模块"""
    
    def __init__(self, output_channels: int):
        super(Decoder, self).__init__()
        
        # 上采样块
        self.up1 = UpBlock(512, 512)
        self.up2 = UpBlock(512, 256)
        self.up3 = UpBlock(256, 128)
        self.up4 = UpBlock(128, 64)
        
        # 最终输出层
        self.final_conv = nn.Conv2d(64, output_channels, kernel_size=7, stride=1, padding=3)
        self.tanh = nn.Tanh()
        
    def forward(self, x: torch.Tensor, skip_features: list) -> torch.Tensor:
        """解码过程"""
        # 上采样（包含跳跃连接）
        x = self.up1(x, skip_features[3] if len(skip_features) > 3 else None)
        x = self.up2(x, skip_features[2] if len(skip_features) > 2 else None)
        x = self.up3(x, skip_features[1] if len(skip_features) > 1 else None)
        x = self.up4(x, skip_features[0] if len(skip_features) > 0 else None)
        
        # 最终输出
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x


class DownBlock(nn.Module):
    """下采样块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(DownBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x


class UpBlock(nn.Module):
    """上采样块"""
    
    def __init__(self, in_channels: int, out_channels: int):
        super(UpBlock, self).__init__()
        
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor, skip_feature: torch.Tensor = None) -> torch.Tensor:
        # 上采样
        x = self.up(x)
        
        # 跳跃连接
        if skip_feature is not None:
            x = torch.cat([x, skip_feature], dim=1)
        
        # 卷积处理
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        
        return x


class SelfAttention(nn.Module):
    """自注意力机制"""
    
    def __init__(self, channels: int):
        super(SelfAttention, self).__init__()
        
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = x.size()
        
        # 生成Q, K, V
        query = self.query(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key(x).view(batch_size, -1, height * width)
        value = self.value(x).view(batch_size, -1, height * width)
        
        # 计算注意力权重
        attention = torch.bmm(query, key)
        attention = F.softmax(attention, dim=-1)
        
        # 应用注意力
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        # 残差连接
        out = self.gamma * out + x
        
        return out


class SkipConnections:
    """跳跃连接管理器"""
    
    def __init__(self):
        self.features = []
        
    def add_feature(self, feature: torch.Tensor):
        """添加特征"""
        self.features.append(feature)
        
    def get_features(self) -> list:
        """获取所有特征"""
        return self.features
        
    def clear(self):
        """清空特征"""
        self.features.clear()


def create_defakehop_model(input_channels: int = 3, output_channels: int = 3) -> DefakeHopModel:
    """
    创建DefakeHop模型实例
    
    Args:
        input_channels: 输入通道数
        output_channels: 输出通道数
        
    Returns:
        DefakeHopModel: 模型实例
    """
    return DefakeHopModel(input_channels, output_channels)


if __name__ == "__main__":
    # 测试模型
    model = create_defakehop_model()
    
    # 创建测试输入
    test_input = torch.randn(1, 3, 256, 256)
    
    # 前向传播
    with torch.no_grad():
        output = model(test_input)
    
    print(f"输入形状: {test_input.shape}")
    print(f"输出形状: {output.shape}")
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
