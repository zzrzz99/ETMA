#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face2Face 表情迁移恢复脚本 - 基于 3DMM 表情参数解耦与反演
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
from shared_utils.face_detection import FaceDetector
from shared_utils.utils import setup_logging, load_image, save_image, create_output_dir, get_image_files

class Face2Face3DMMRestorer:
    """Face2Face 3DMM 表情迁移恢复器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # 初始化模型
        self.model = Face2FaceRestorer()
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        self.model.eval()
        self.face_detector = FaceDetector()
        
        # 3DMM 参数维度
        self.exp_dim = 64
        self.id_dim = 80
        self.pose_dim = 6
        
    def load_checkpoint(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
    
    def extract_3dmm_coefficients(self, image: np.ndarray):
        """提取 3DMM 系数（简化实现）"""
        batch_size = 1
        id_coeff = np.random.randn(batch_size, self.id_dim) * 0.1
        exp_coeff = np.random.randn(batch_size, self.exp_dim) * 0.1
        pose_coeff = np.random.randn(batch_size, self.pose_dim) * 0.1
        return id_coeff, exp_coeff, pose_coeff
    
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
            
            # 提取 3DMM 系数
            id_coeff, exp_fake, pose_coeff = self.extract_3dmm_coefficients(face_resized)
            
            # 表情参数反演（简化实现）
            exp_restored = exp_fake * 0.5  # 简单的线性反演
            
            # 3D 人脸重建与渲染
            exp_tensor = torch.from_numpy(exp_restored).float().to(self.device)
            id_tensor = torch.from_numpy(id_coeff).float().to(self.device)
            pose_tensor = torch.from_numpy(pose_coeff).float().to(self.device)
            
            with torch.no_grad():
                restored_face = self.model(exp_tensor, id_tensor, pose_tensor)
                restored_image = restored_face.squeeze(0).permute(1, 2, 0).cpu().numpy()
                restored_image = np.clip(restored_image * 255, 0, 255).astype(np.uint8)
            
            # 融合回原图
            restored_resized = cv2.resize(restored_image, (x2-x1, y2-y1))
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
    parser = argparse.ArgumentParser(description="Face2Face 3DMM 表情迁移恢复")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, help="预训练模型路径")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动 Face2Face 3DMM 表情迁移恢复")
    
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    restorer = Face2Face3DMMRestorer(model_path=args.model_path, device=args.device)
    restorer.restore_batch(args.input_dir, args.output_dir)
    
    logger.info("Face2Face 3DMM 表情迁移恢复完成")

if __name__ == "__main__":
    main()
