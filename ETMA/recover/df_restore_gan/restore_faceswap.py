#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FaceSwap 恢复脚本 - DeepFaceLab-Restore + GAN 反演
基于 3D 几何先验恢复原始身份纹理与光照一致性
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

from .model import FaceSwapRestorer
from shared_utils.face_detection import FaceDetector
from shared_utils.utils import setup_logging, load_image, save_image, create_output_dir, get_image_files

class FaceSwapDeepLabRestorer:
    """FaceSwap DeepLab-Restore + GAN 反演恢复器"""
    
    def __init__(self, model_path: str = None, device: str = "auto"):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() and device != "cpu" else "cpu")
        
        # 初始化模型
        self.model = FaceSwapRestorer()
        self.model.to(self.device)
        
        if model_path and os.path.exists(model_path):
            self.load_checkpoint(model_path)
        
        self.model.eval()
        self.face_detector = FaceDetector()
        
        # 3DMM 参数维度
        self.id_dim = 512
        self.tex_dim = 256
        self.light_dim = 27
        
    def load_checkpoint(self, model_path: str):
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            self.logger.info("模型加载成功")
        except Exception as e:
            self.logger.error(f"模型加载失败: {e}")
    
    def load_3dmm_coefficients(self, id_path: str, tex_path: str, light_path: str):
        """加载 3DMM 系数"""
        try:
            id_coeff = np.load(id_path)
            tex_coeff = np.load(tex_path)
            light_coeff = np.load(light_path)
            
            self.logger.info("成功加载 3DMM 系数")
            return id_coeff, tex_coeff, light_coeff
            
        except Exception as e:
            self.logger.error(f"加载 3DMM 系数失败: {e}")
            # 生成随机系数作为备选
            batch_size = 1
            id_coeff = np.random.randn(batch_size, self.id_dim) * 0.1
            tex_coeff = np.random.randn(batch_size, self.tex_dim) * 0.1
            light_coeff = np.random.randn(batch_size, self.light_dim) * 0.1
            return id_coeff, tex_coeff, light_coeff
    
    def restore_single_image(self, image_path: str, output_path: str, id_params: np.ndarray, 
                           tex_params: np.ndarray, light_params: np.ndarray) -> bool:
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
            face_tensor = torch.from_numpy(face_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            id_tensor = torch.from_numpy(id_params).float().to(self.device)
            tex_tensor = torch.from_numpy(tex_params).float().to(self.device)
            light_tensor = torch.from_numpy(light_params).float().to(self.device)
            
            with torch.no_grad():
                restored_face, latent_code = self.model(face_tensor, id_tensor, tex_tensor, light_tensor)
                
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
    
    def restore_batch(self, input_dir: str, output_dir: str, id_path: str, tex_path: str, light_path: str):
        """批量恢复图像"""
        if not create_output_dir(output_dir):
            return
        
        # 加载 3DMM 系数
        id_params, tex_params, light_params = self.load_3dmm_coefficients(id_path, tex_path, light_path)
        
        image_files = get_image_files(input_dir)
        if not image_files:
            return
        
        self.logger.info(f"开始批量恢复 {len(image_files)} 个图像")
        
        for image_file in image_files:
            rel_path = os.path.relpath(image_file, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.restore_single_image(image_file, output_path, id_params, tex_params, light_params)

def main():
    parser = argparse.ArgumentParser(description="FaceSwap DeepLab-Restore + GAN 反演")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--id_npy", type=str, required=True, help="身份系数文件路径")
    parser.add_argument("--tex_npy", type=str, required=True, help="纹理系数文件路径")
    parser.add_argument("--light_npy", type=str, required=True, help="光照系数文件路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    parser.add_argument("--model_path", type=str, help="预训练模型路径")
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动 FaceSwap DeepLab-Restore + GAN 反演")
    
    # 检查输入文件
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    # 创建恢复器
    restorer = FaceSwapDeepLabRestorer(model_path=args.model_path, device=args.device)
    
    # 执行恢复
    restorer.restore_batch(args.input_dir, args.output_dir, args.id_npy, args.tex_npy, args.light_npy)
    
    logger.info("FaceSwap DeepLab-Restore + GAN 反演完成")

if __name__ == "__main__":
    main()
