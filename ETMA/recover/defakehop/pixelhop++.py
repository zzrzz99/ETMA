#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PixelHop++ 特征提取模块

实现基于 PixelHop++ 的潜在编码提取，用于 DeepFake 检测和恢复
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
import logging
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PixelHopUnit:
    """PixelHop++ 单元实现"""
    
    def __init__(self, depth: int = 3, TH1: float = 0.005, TH2: float = 0.001):
        """
        初始化 PixelHop++ 单元
        
        Args:
            depth: Hop 层数
            TH1: 第一阈值（能量保留）
            TH2: 第二阈值（特征选择）
        """
        self.depth = depth
        self.TH1 = TH1
        self.TH2 = TH2
        self.logger = logging.getLogger(__name__)
        
        # 存储每层的变换参数
        self.transformers = []
        self.scalers = []
        
    def fit(self, patches: np.ndarray) -> 'PixelHopUnit':
        """
        训练 PixelHop++ 变换器
        
        Args:
            patches: 输入图像块 [N, H, W, C]
            
        Returns:
            self: 训练后的变换器
        """
        self.logger.info(f"开始训练 PixelHop++ 变换器，输入形状: {patches.shape}")
        
        current_data = patches
        
        for hop in range(self.depth):
            self.logger.info(f"训练 Hop {hop + 1}")
            
            # 标准化
            scaler = StandardScaler()
            current_data_flat = current_data.reshape(current_data.shape[0], -1)
            current_data_scaled = scaler.fit_transform(current_data_flat)
            self.scalers.append(scaler)
            
            # Saab 变换（简化版本，使用 PCA 近似）
            pca = PCA(n_components=0.9)  # 保留90%能量
            current_data_transformed = pca.fit_transform(current_data_scaled)
            
            # 存储变换器
            self.transformers.append(pca)
            
            # 重塑数据用于下一层
            if hop < self.depth - 1:
                # 计算新的空间维度
                spatial_dim = int(np.sqrt(current_data_transformed.shape[1]))
                current_data = current_data_transformed.reshape(
                    current_data_transformed.shape[0], 
                    spatial_dim, 
                    spatial_dim, 
                    -1
                )
                
                # 空间下采样
                current_data = self._spatial_downsample(current_data)
        
        self.logger.info("PixelHop++ 训练完成")
        return self
    
    def transform(self, patches: np.ndarray) -> np.ndarray:
        """
        提取特征（潜在编码）
        
        Args:
            patches: 输入图像块 [N, H, W, C]
            
        Returns:
            np.ndarray: 潜在编码 [N, features]
        """
        if not self.transformers:
            raise ValueError("请先调用 fit() 方法训练变换器")
        
        self.logger.info(f"开始特征提取，输入形状: {patches.shape}")
        
        current_data = patches
        all_features = []
        
        for hop in range(self.depth):
            # 标准化
            current_data_flat = current_data.reshape(current_data.shape[0], -1)
            current_data_scaled = self.scalers[hop].transform(current_data_flat)
            
            # 变换
            current_data_transformed = self.transformers[hop].transform(current_data_scaled)
            
            # 收集特征
            all_features.append(current_data_transformed)
            
            # 准备下一层
            if hop < self.depth - 1:
                spatial_dim = int(np.sqrt(current_data_transformed.shape[1]))
                current_data = current_data_transformed.reshape(
                    current_data_transformed.shape[0], 
                    spatial_dim, 
                    spatial_dim, 
                    -1
                )
                current_data = self._spatial_downsample(current_data)
        
        # 拼接所有层的特征
        final_features = np.concatenate(all_features, axis=1)
        
        self.logger.info(f"特征提取完成，输出形状: {final_features.shape}")
        return final_features
    
    def _spatial_downsample(self, data: np.ndarray) -> np.ndarray:
        """
        空间下采样
        
        Args:
            data: 输入数据 [N, H, W, C]
            
        Returns:
            np.ndarray: 下采样后的数据
        """
        N, H, W, C = data.shape
        
        # 简单的平均池化
        new_H, new_W = H // 2, W // 2
        downsampled = np.zeros((N, new_H, new_W, C))
        
        for i in range(new_H):
            for j in range(new_W):
                downsampled[:, i, j, :] = np.mean(
                    data[:, i*2:(i+1)*2, j*2:(j+1)*2, :], 
                    axis=(1, 2)
                )
        
        return downsampled


class DefakeHopFeatureExtractor:
    """DefakeHop 特征提取器"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        初始化特征提取器
        
        Args:
            model_path: 预训练模型路径
        """
        self.logger = logging.getLogger(__name__)
        
        # 创建 PixelHop++ 单元
        self.pixelhop = PixelHopUnit(
            depth=3,
            TH1=0.005,
            TH2=0.001
        )
        
        # 伪造痕迹子空间
        self.forged_subspace = None
        
        if model_path:
            self.load_model(model_path)
    
    def extract_patch_features(self, patches: np.ndarray) -> np.ndarray:
        """
        提取图像块特征
        
        Args:
            patches: 输入图像块 [N, 32, 32, 3]
            
        Returns:
            np.ndarray: 特征向量 [N, 5625]
        """
        # 确保输入尺寸正确
        if patches.shape[1:] != (32, 32, 3):
            raise ValueError(f"期望输入尺寸 (32, 32, 3)，实际: {patches.shape[1:]}")
        
        # 提取特征
        features = self.pixelhop.transform(patches)
        
        # 确保输出维度正确
        if features.shape[1] != 5625:
            self.logger.warning(f"特征维度不匹配，期望 5625，实际: {features.shape[1]}")
            # 调整到目标维度
            if features.shape[1] > 5625:
                features = features[:, :5625]
            else:
                # 填充到目标维度
                padding = np.zeros((features.shape[0], 5625 - features.shape[1]))
                features = np.concatenate([features, padding], axis=1)
        
        return features
    
    def remove_forged_artifacts(self, features: np.ndarray) -> np.ndarray:
        """
        去除伪造痕迹
        
        Args:
            features: 输入特征 [N, 5625]
            
        Returns:
            np.ndarray: 清理后的特征 [N, 5625]
        """
        if self.forged_subspace is None:
            self.logger.warning("伪造子空间未加载，跳过痕迹去除")
            return features
        
        # 反演公式: z_rec = z_fake - B · (B^T · z_fake)
        B = self.forged_subspace
        
        # 计算投影
        projection = np.dot(features, B)
        reconstructed = np.dot(projection, B.T)
        
        # 去除伪造痕迹
        cleaned_features = features - reconstructed
        
        return cleaned_features
    
    def load_model(self, model_path: str) -> bool:
        """
        加载预训练模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            bool: 是否加载成功
        """
        try:
            # 加载 PixelHop++ 参数
            model_data = np.load(model_path, allow_pickle=True)
            
            # 恢复变换器
            self.pixelhop.transformers = model_data['transformers'].tolist()
            self.pixelhop.scalers = model_data['scalers'].tolist()
            
            # 加载伪造子空间
            if 'forged_subspace' in model_data:
                self.forged_subspace = model_data['forged_subspace']
            
            self.logger.info(f"成功加载模型: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
            return False
    
    def save_model(self, model_path: str) -> bool:
        """
        保存模型
        
        Args:
            model_path: 保存路径
            
        Returns:
            bool: 是否保存成功
        """
        try:
            model_data = {
                'transformers': np.array(self.pixelhop.transformers, dtype=object),
                'scalers': np.array(self.pixelhop.scalers, dtype=object)
            }
            
            if self.forged_subspace is not None:
                model_data['forged_subspace'] = self.forged_subspace
            
            np.savez(model_path, **model_data)
            self.logger.info(f"成功保存模型: {model_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存模型失败: {e}")
            return False
    
    def train_forged_subspace(self, fake_features: np.ndarray, real_features: np.ndarray) -> bool:
        """
        训练伪造痕迹子空间
        
        Args:
            fake_features: 伪造图像特征 [N, 5625]
            real_features: 真实图像特征 [N, 5625]
            
        Returns:
            bool: 是否训练成功
        """
        try:
            # 计算残差
            residuals = fake_features - real_features
            
            # PCA 提取主成分
            pca = PCA(n_components=64)
            pca.fit(residuals)
            
            # 保存伪造子空间
            self.forged_subspace = pca.components_.T
            
            self.logger.info(f"成功训练伪造子空间，维度: {self.forged_subspace.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"训练伪造子空间失败: {e}")
            return False


def create_feature_extractor(model_path: Optional[str] = None) -> DefakeHopFeatureExtractor:
    """
    创建特征提取器实例
    
    Args:
        model_path: 预训练模型路径
        
    Returns:
        DefakeHopFeatureExtractor: 特征提取器实例
    """
    return DefakeHopFeatureExtractor(model_path)


if __name__ == "__main__":
    # 测试特征提取器
    logging.basicConfig(level=logging.INFO)
    
    # 创建随机测试数据
    test_patches = np.random.randn(10, 32, 32, 3).astype(np.float32)
    
    # 创建特征提取器
    extractor = create_feature_extractor()
    
    # 训练（如果没有预训练模型）
    if not extractor.pixelhop.transformers:
        print("训练 PixelHop++ 变换器...")
        extractor.pixelhop.fit(test_patches)
    
    # 提取特征
    features = extractor.extract_patch_features(test_patches)
    print(f"特征提取完成，输出形状: {features.shape}")
    
    # 保存模型
    extractor.save_model("test_model.npz")
    print("模型保存完成")





