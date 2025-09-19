#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face2Face 3DMM 表情参数反演脚本
"""

import os
import sys
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
import torch

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from .model import Face2FaceRestorer
from shared_utils.utils import setup_logging

class Face2Face3DMMReverter:
    """Face2Face 3DMM 表情参数反演器"""
    
    def __init__(self, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # 初始化模型
        self.model = Face2FaceRestorer()
        self.model.to(self.device)
        self.model.eval()
        
        # 3DMM 参数维度
        self.exp_dim = 64
        self.id_dim = 80
        self.pose_dim = 6
        
    def load_3dmm_coefficients(self, exp_path: str, id_path: str, pose_path: str):
        """加载 3DMM 系数"""
        try:
            exp_coeff = np.load(exp_path)
            id_coeff = np.load(id_path)
            pose_coeff = np.load(pose_path)
            
            self.logger.info(f"成功加载 3DMM 系数")
            return exp_coeff, id_coeff, pose_coeff
            
        except Exception as e:
            self.logger.error(f"加载 3DMM 系数失败: {e}")
            # 生成随机系数作为备选
            batch_size = 1
            exp_coeff = np.random.randn(batch_size, self.exp_dim) * 0.1
            id_coeff = np.random.randn(batch_size, self.id_dim) * 0.1
            pose_coeff = np.random.randn(batch_size, self.pose_dim) * 0.1
            return exp_coeff, id_coeff, pose_coeff
    
    def invert_expression_parameters(self, exp_fake: np.ndarray, exp_target: np.ndarray = None) -> np.ndarray:
        """表情参数反演"""
        if exp_target is None:
            exp_target = np.zeros_like(exp_fake)
        
        # 表情残差反演：βʳ = βᵗ + Δβ
        exp_residual = exp_fake - exp_target
        exp_restored = exp_target + 0.5 * exp_residual
        
        return exp_restored
    
    def reconstruct_3d_face(self, exp_coeff: np.ndarray, id_coeff: np.ndarray, 
                           pose_coeff: np.ndarray) -> np.ndarray:
        """3D 人脸重建与渲染"""
        # 转换为 PyTorch 张量
        exp_tensor = torch.from_numpy(exp_coeff).float().to(self.device)
        id_tensor = torch.from_numpy(id_coeff).float().to(self.device)
        pose_tensor = torch.from_numpy(pose_coeff).float().to(self.device)
        
        # 模型推理
        with torch.no_grad():
            restored_face = self.model(exp_tensor, id_tensor, pose_tensor)
            
            # 转换回 numpy 数组
            restored_image = restored_face.squeeze(0).permute(1, 2, 0).cpu().numpy()
            restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)
        
        return restored_image
    
    def process_video(self, input_video: str, exp_path: str, id_path: str, 
                     pose_path: str, output_dir: str, mask_path: str = None):
        """处理整个视频"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载 3DMM 系数
        exp_coeff, id_coeff, pose_coeff = self.load_3dmm_coefficients(exp_path, id_path, pose_path)
        
        # 表情参数反演
        self.logger.info("开始表情参数反演...")
        exp_restored = self.invert_expression_parameters(exp_coeff)
        
        # 3D 人脸重建与渲染
        self.logger.info("开始 3D 人脸重建...")
        restored_face = self.reconstruct_3d_face(exp_restored, id_coeff, pose_coeff)
        
        # 保存结果
        output_path = os.path.join(output_dir, "restored_face.png")
        cv2.imwrite(output_path, cv2.cvtColor(restored_face, cv2.COLOR_RGB2BGR))
        
        # 保存恢复后的表情系数
        exp_restored_path = os.path.join(output_dir, "exp_restored.npy")
        np.save(exp_restored_path, exp_restored)
        
        self.logger.info(f"处理完成，结果保存在: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Face2Face 3DMM 表情参数反演")
    parser.add_argument("--input_video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--exp_path", type=str, required=True, help="表情系数文件路径")
    parser.add_argument("--id_path", type=str, required=True, help="身份系数文件路径")
    parser.add_argument("--pose_path", type=str, required=True, help="姿态系数文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--mask_path", type=str, help="嘴部掩码路径（可选）")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动 Face2Face 3DMM 表情参数反演")
    
    # 检查输入文件
    if not os.path.exists(args.input_video):
        logger.error(f"输入视频不存在: {args.input_video}")
        return
    
    # 创建反演器
    reverter = Face2Face3DMMReverter(device=args.device)
    
    # 处理视频
    reverter.process_video(
        input_video=args.input_video,
        exp_path=args.exp_path,
        id_path=args.id_path,
        pose_path=args.pose_path,
        output_dir=args.output_dir,
        mask_path=args.mask_path
    )
    
    logger.info("Face2Face 3DMM 表情参数反演完成")

if __name__ == "__main__":
    main()
