#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFake 恢复主脚本

基于 DefakeHop 的 DeepFake 检测和恢复实现
"""

import argparse
import os
import sys
import logging
import time
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np
from PIL import Image

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.utils import setup_logging, load_image, save_image, create_output_dir
from .pixelhop import DefakeHopFeatureExtractor
from .decoder import load_decoder


class DeepFakeRestorer:
    """DeepFake 恢复器"""
    
    def __init__(self, model_dir: str, device: str = "auto"):
        """
        初始化恢复器
        
        Args:
            model_dir: 模型目录路径
            device: 计算设备
        """
        self.logger = logging.getLogger(__name__)
        self.device = device
        
        # 加载模型
        self.feature_extractor = None
        self.decoder = None
        self.load_models(model_dir)
        
        # 伪造痕迹子空间
        self.forged_subspace = None
        self.load_forged_subspace(model_dir)
    
    def load_models(self, model_dir: str):
        """加载模型"""
        try:
            # 加载特征提取器
            feature_model_path = os.path.join(model_dir, "defakehop_features.npz")
            if os.path.exists(feature_model_path):
                self.feature_extractor = DefakeHopFeatureExtractor(feature_model_path)
                self.logger.info("成功加载特征提取器")
            else:
                self.logger.warning("特征提取器模型不存在，将使用随机初始化")
                self.feature_extractor = DefakeHopFeatureExtractor()
            
            # 加载解码器
            decoder_path = os.path.join(model_dir, "dehop_decoder.pth")
            if os.path.exists(decoder_path):
                self.decoder = load_decoder(decoder_path, self.device)
                self.logger.info("成功加载解码器")
            else:
                self.logger.warning("解码器模型不存在，将使用随机初始化")
                self.decoder = load_decoder(device=self.device)
                
        except Exception as e:
            self.logger.error(f"加载模型失败: {e}")
    
    def load_forged_subspace(self, model_dir: str):
        """加载伪造痕迹子空间"""
        try:
            subspace_path = os.path.join(model_dir, "forged_subspace.npz")
            if os.path.exists(subspace_path):
                data = np.load(subspace_path)
                self.forged_subspace = data['subspace']
                self.logger.info(f"成功加载伪造子空间，维度: {self.forged_subspace.shape}")
            else:
                self.logger.warning("伪造子空间文件不存在")
        except Exception as e:
            self.logger.error(f"加载伪造子空间失败: {e}")
    
    def extract_face_patches(self, image: np.ndarray) -> List[np.ndarray]:
        """
        提取人脸区域的小 patch
        
        Args:
            image: 输入图像
            
        Returns:
            List[np.ndarray]: 三个区域的 patch 列表
        """
        # 简化的区域提取（实际应用中应使用 OpenFace2 提取 landmarks）
        h, w = image.shape[:2]
        
        # 左眼中心 (约 1/3 高度，1/4 宽度)
        left_eye_x = w // 4
        left_eye_y = h // 3
        
        # 右眼中心 (约 1/3 高度，3/4 宽度)
        right_eye_x = 3 * w // 4
        right_eye_y = h // 3
        
        # 嘴部中心 (约 2/3 高度，1/2 宽度)
        mouth_x = w // 2
        mouth_y = 2 * h // 3
        
        # 提取 32x32 的 patch
        patch_size = 32
        patches = []
        
        for center_x, center_y in [(left_eye_x, left_eye_y), (right_eye_x, right_eye_y), (mouth_x, mouth_y)]:
            # 计算 patch 边界
            x1 = max(0, center_x - patch_size // 2)
            y1 = max(0, center_y - patch_size // 2)
            x2 = min(w, x1 + patch_size)
            y2 = min(h, y1 + patch_size)
            
            # 提取 patch
            patch = image[y1:y2, x1:x2]
            
            # 调整到目标尺寸
            if patch.shape[:2] != (patch_size, patch_size):
                patch = cv2.resize(patch, (patch_size, patch_size))
            
            patches.append(patch)
        
        return patches
    
    def restore_single_image(self, image_path: str, output_path: str) -> bool:
        """
        恢复单张图像
        
        Args:
            image_path: 输入图像路径
            output_path: 输出图像路径
            
        Returns:
            bool: 是否成功
        """
        try:
            # 加载图像
            image = load_image(image_path)
            if image is None:
                return False
            
            # 调整图像尺寸到 128x128
            image = cv2.resize(image, (128, 128))
            
            # 提取人脸 patch
            patches = self.extract_face_patches(image)
            
            if len(patches) != 3:
                self.logger.error(f"提取的 patch 数量不正确: {len(patches)}")
                return False
            
            # 转换为 numpy 数组
            patches_array = np.array(patches)  # [3, 32, 32, 3]
            
            # 提取特征
            features = self.feature_extractor.extract_patch_features(patches_array)
            
            # 去除伪造痕迹
            if self.forged_subspace is not None:
                features = self.feature_extractor.remove_forged_artifacts(features)
            
            # 解码重建图像
            restored_patches = self.decoder.decode(features)
            
            # 将恢复的 patch 融合回原图
            restored_image = self.fuse_patches_to_image(image, restored_patches)
            
            # 保存结果
            save_image(restored_image, output_path)
            
            self.logger.info(f"成功恢复图像: {image_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复图像失败 {image_path}: {e}")
            return False
    
    def fuse_patches_to_image(self, original_image: np.ndarray, restored_patches: np.ndarray) -> np.ndarray:
        """
        将恢复的 patch 融合回原图
        
        Args:
            original_image: 原始图像
            restored_patches: 恢复的 patch
            
        Returns:
            np.ndarray: 融合后的图像
        """
        # 创建输出图像副本
        output_image = original_image.copy()
        h, w = original_image.shape[:2]
        
        # 定义 patch 中心位置（与提取时一致）
        patch_size = 32
        centers = [
            (w // 4, h // 3),      # 左眼
            (3 * w // 4, h // 3),  # 右眼
            (w // 2, 2 * h // 3)   # 嘴部
        ]
        
        # 融合每个 patch
        for i, (center_x, center_y) in enumerate(centers):
            if i >= len(restored_patches):
                break
                
            # 计算 patch 边界
            x1 = max(0, center_x - patch_size // 2)
            y1 = max(0, center_y - patch_size // 2)
            x2 = min(w, x1 + patch_size)
            y2 = min(h, y1 + patch_size)
            
            # 获取当前 patch
            current_patch = restored_patches[i]
            
            # 调整 patch 尺寸以匹配目标区域
            target_h, target_w = y2 - y1, x2 - x1
            if current_patch.shape[:2] != (target_h, target_w):
                current_patch = cv2.resize(current_patch, (target_w, target_h))
            
            # 简单的 alpha 融合
            alpha = 0.7
            output_image[y1:y2, x1:x2] = (
                alpha * current_patch + (1 - alpha) * original_image[y1:y2, x1:x2]
            ).astype(np.uint8)
        
        return output_image
    
    def restore_batch(self, input_dir: str, output_dir: str) -> dict:
        """
        批量恢复图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            dict: 处理结果统计
        """
        # 创建输出目录
        create_output_dir(output_dir)
        
        # 获取图像文件列表
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = []
        
        input_path = Path(input_dir)
        for file_path in input_path.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in image_extensions:
                image_files.append(str(file_path))
        
        if not image_files:
            self.logger.warning(f"在 {input_dir} 中未找到图像文件")
            return {"total": 0, "success": 0, "failed": 0}
        
        self.logger.info(f"开始批量处理 {len(image_files)} 张图像")
        
        results = {"total": len(image_files), "success": 0, "failed": 0}
        
        for image_file in image_files:
            # 生成输出文件名
            base_name = Path(image_file).stem
            output_name = f"{base_name}_restored.png"
            output_path = os.path.join(output_dir, output_name)
            
            if self.restore_single_image(image_file, output_path):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        self.logger.info(f"批量处理完成: {results}")
        return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DeepFake 恢复 - 基于 DefakeHop")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model_dir", default="checkpoints/", help="模型目录")
    parser.add_argument("--device", default="auto", choices=["cpu", "cuda", "auto"], help="计算设备")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 1
    
    # 检查模型目录
    if not os.path.exists(args.model_dir):
        logger.warning(f"模型目录不存在: {args.model_dir}，将使用随机初始化")
    
    start_time = time.time()
    
    # 创建恢复器
    restorer = DeepFakeRestorer(args.model_dir, args.device)
    
    # 执行恢复
    results = restorer.restore_batch(args.input_dir, args.output_dir)
    
    # 输出结果
    total_time = time.time() - start_time
    logger.info(f"处理完成: 总计 {results['total']} 张，成功 {results['success']} 张，失败 {results['failed']} 张")
    logger.info(f"总耗时: {total_time:.2f}秒")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())





