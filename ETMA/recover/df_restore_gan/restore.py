#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepFaceLab-Restore + GAN Inversion 恢复脚本
针对 FaceSwap 伪造视频进行恢复，使用 GAN 反演技术
"""

import os
import sys
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from .model import FaceSwapModel
from shared_utils.face_detection import FaceDetector
from shared_utils.utils import setup_logging, load_image, save_image, create_output_dir, get_image_files

class FaceSwapRestorer:
    """FaceSwap 恢复器（基于 GAN Inversion）"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # 初始化模型
        self.model = FaceSwapModel()
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        self.model.eval()
        self.face_detector = FaceDetector()
        
        # GAN 反演参数
        self.latent_dim = 512
        self.num_iterations = 1000
        self.learning_rate = 0.01
        
    def load_checkpoint(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
    
    def gan_inversion(self, target_image: np.ndarray) -> np.ndarray:
        """
        执行 GAN 反演
        
        Args:
            target_image: 目标图像
            
        Returns:
            反演后的图像
        """
        # 转换为 PyTorch 张量
        target_tensor = torch.from_numpy(target_image).float().permute(2, 0, 1).unsqueeze(0) / 255.0
        target_tensor = target_tensor.to(self.device)
        
        # 初始化潜在向量
        latent = torch.randn(1, self.latent_dim, device=self.device, requires_grad=True)
        optimizer = torch.optim.Adam([latent], lr=self.learning_rate)
        
        # 迭代优化
        for i in range(self.num_iterations):
            optimizer.zero_grad()
            
            # 生成图像
            generated = self.model.gan_encoder(latent)
            
            # 计算损失
            loss = nn.MSELoss()(generated, target_tensor)
            
            # 添加正则化
            reg_loss = torch.norm(latent, p=2)
            total_loss = loss + 0.01 * reg_loss
            
            total_loss.backward()
            optimizer.step()
            
            if i % 100 == 0:
                self.logger.debug(f"迭代 {i}, 损失: {total_loss.item():.6f}")
        
        # 生成最终结果
        with torch.no_grad():
            restored = self.model.restore_net(generated)
            restored_image = restored.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)
        
        return restored_image
    
    def restore_single_image(self, image_path: str, output_path: str) -> bool:
        try:
            image = load_image(image_path)
            if image is None:
                return False
            
            # 检测人脸
            faces = self.face_detector.detect_faces(image)
            if not faces:
                return save_image(image, output_path)
            
            # 使用最大人脸
            face_bbox = max(faces, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))
            x1, y1, x2, y2 = face_bbox
            
            # 裁剪人脸
            face_region = image[y1:y2, x1:x2]
            face_resized = cv2.resize(face_region, (256, 256))
            
            # GAN 反演恢复
            restored_face = self.gan_inversion(face_resized)
            
            # 融合回原图
            restored_resized = cv2.resize(restored_face, (x2-x1, y2-y1))
            result = image.copy()
            result[y1:y2, x1:x2] = restored_resized
            
            return save_image(result, output_path)
            
        except Exception as e:
            self.logger.error(f"处理图像失败: {e}")
            return False
    
    def restore_batch(self, input_dir: str, output_dir: str):
        if not create_output_dir(output_dir):
            return
        
        image_files = get_image_files(input_dir)
        if not image_files:
            return
        
        for image_file in image_files:
            rel_path = os.path.relpath(image_file, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            self.restore_single_image(image_file, output_path)

def main():
    parser = argparse.ArgumentParser(description="DeepFaceLab-Restore + GAN Inversion 恢复")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, help="预训练模型路径")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    parser.add_argument("--iterations", type=int, default=1000, help="GAN 反演迭代次数")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动 DeepFaceLab-Restore + GAN Inversion 恢复")
    
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    restorer = FaceSwapRestorer(model_path=args.model_path, device=args.device)
    restorer.num_iterations = args.iterations
    restorer.restore_batch(args.input_dir, args.output_dir)
    
    logger.info("DeepFaceLab-Restore + GAN Inversion 恢复完成")

if __name__ == "__main__":
    main()
