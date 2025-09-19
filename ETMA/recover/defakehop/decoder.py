#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
反 PixelHop++ 解码器

将潜在编码映射回像素空间，重建原始人脸图像
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
import logging


class DehopDecoder(nn.Module):
    """反 PixelHop++ 解码器"""
    
    def __init__(self, latent_dim: int = 5625, output_size: int = 128):
        """
        初始化解码器
        
        Args:
            latent_dim: 潜在编码维度
            output_size: 输出图像尺寸
        """
        super(DehopDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.output_size = output_size
        
        # 计算中间特征图尺寸
        self.feature_size = output_size // 8  # 从 8x8 开始上采样
        
        # 潜在编码到特征图的映射
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, 512 * self.feature_size * self.feature_size),
            nn.BatchNorm1d(512 * self.feature_size * self.feature_size),
            nn.ReLU(inplace=True)
        )
        
        # 上采样网络
        self.upsample_net = nn.ModuleList([
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True)
            ),
            # 64x64 -> 128x128
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 最终输出层
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )
        
    def forward(self, latent_codes: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            latent_codes: 潜在编码 [B, latent_dim]
            
        Returns:
            torch.Tensor: 重建图像 [B, 3, H, W]
        """
        batch_size = latent_codes.size(0)
        
        # 潜在编码到特征图
        features = self.latent_to_features(latent_codes)
        features = features.view(batch_size, 512, self.feature_size, self.feature_size)
        
        # 逐步上采样
        for upsample_layer in self.upsample_net:
            features = upsample_layer(features)
        
        # 最终输出
        output = self.final_conv(features)
        
        return output


class DefakeHopDecoder:
    """DefakeHop 解码器包装类"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化解码器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备
        """
        self.logger = logging.getLogger(__name__)
        self.device = self._setup_device(device)
        
        # 创建解码器模型
        self.decoder = DehopDecoder()
        self.decoder.to(self.device)
        
        # 加载预训练权重
        if model_path:
            self.load_checkpoint(model_path)
        
        self.decoder.eval()
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，使用CPU")
            device = "cpu"
            
        return torch.device(device)
    
    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        加载预训练权重
        
        Args:
            checkpoint_path: 权重文件路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            
            if 'model_state_dict' in checkpoint:
                self.decoder.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.decoder.load_state_dict(checkpoint)
            
            self.logger.info(f"成功加载解码器权重: {checkpoint_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载解码器权重失败: {e}")
            return False
    
    def decode(self, latent_codes: np.ndarray) -> np.ndarray:
        """
        解码潜在编码
        
        Args:
            latent_codes: 潜在编码 [N, 5625]
            
        Returns:
            np.ndarray: 重建图像 [N, 128, 128, 3]
        """
        # 转换为 tensor
        latent_tensor = torch.from_numpy(latent_codes).float().to(self.device)
        
        with torch.no_grad():
            # 解码
            output_tensor = self.decoder(latent_tensor)
            
            # 转换为 numpy
            output_np = output_tensor.cpu().numpy()
            
            # 调整通道顺序 [B, C, H, W] -> [B, H, W, C]
            output_np = np.transpose(output_np, (0, 2, 3, 1))
            
            # 转换到 [0, 255] 范围
            output_np = (output_np + 1) * 127.5  # [-1, 1] -> [0, 255]
            output_np = np.clip(output_np, 0, 255).astype(np.uint8)
        
        return output_np
    
    def decode_single(self, latent_code: np.ndarray) -> np.ndarray:
        """
        解码单个潜在编码
        
        Args:
            latent_code: 单个潜在编码 [5625]
            
        Returns:
            np.ndarray: 重建图像 [128, 128, 3]
        """
        # 添加 batch 维度
        latent_batch = latent_code[np.newaxis, :]
        
        # 解码
        output_batch = self.decode(latent_batch)
        
        # 移除 batch 维度
        return output_batch[0]


def create_decoder(model_path: Optional[str] = None, device: str = "auto") -> DefakeHopDecoder:
    """
    创建解码器实例
    
    Args:
        model_path: 预训练模型路径
        device: 计算设备
        
    Returns:
        DefakeHopDecoder: 解码器实例
    """
    return DefakeHopDecoder(model_path, device)


def load_decoder(checkpoint_path: str, device: str = "auto") -> DefakeHopDecoder:
    """
    加载预训练解码器
    
    Args:
        checkpoint_path: 权重文件路径
        device: 计算设备
        
    Returns:
        DefakeHopDecoder: 解码器实例
    """
    return create_decoder(checkpoint_path, device)


if __name__ == "__main__":
    # 测试解码器
    logging.basicConfig(level=logging.INFO)
    
    # 创建随机潜在编码
    test_latent = np.random.randn(5, 5625).astype(np.float32)
    
    # 创建解码器
    decoder = create_decoder()
    
    # 测试解码
    output_images = decoder.decode(test_latent)
    print(f"解码完成，输出形状: {output_images.shape}")
    
    # 测试单个解码
    single_output = decoder.decode_single(test_latent[0])
    print(f"单个解码完成，输出形状: {single_output.shape}")





