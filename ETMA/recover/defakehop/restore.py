#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DefakeHop 人脸恢复主脚本

使用方法:
    python restore.py --input_dir ../input/cropped_faces --output_dir ../output/defakehop
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import cv2

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from shared_utils.utils import setup_logging, load_image, save_image
from .model import DefakeHopModel


class DefakeHopRestorer:
    """DefakeHop 人脸恢复器"""
    
    def __init__(self, model_path: Optional[str] = None, device: str = "auto"):
        """
        初始化恢复器
        
        Args:
            model_path: 预训练模型路径
            device: 计算设备 ('cpu', 'cuda', 'auto')
        """
        self.device = self._setup_device(device)
        self.model = self._load_model(model_path)
        self.logger = logging.getLogger(__name__)
        
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            self.logger.warning("CUDA不可用，使用CPU")
            device = "cpu"
            
        return torch.device(device)
    
    def _load_model(self, model_path: Optional[str]) -> DefakeHopModel:
        """加载预训练模型"""
        model = DefakeHopModel()
        
        if model_path and os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"成功加载模型: {model_path}")
            except Exception as e:
                self.logger.warning(f"加载模型失败: {e}，使用随机初始化")
        else:
            self.logger.info("使用随机初始化的模型")
            
        model = model.to(self.device)
        model.eval()
        return model
    
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
                
            # 预处理
            processed_image = self._preprocess(image)
            
            # 模型推理
            with torch.no_grad():
                restored_image = self.model(processed_image)
            
            # 后处理
            final_image = self._postprocess(restored_image)
            
            # 保存结果
            save_image(final_image, output_path)
            
            self.logger.info(f"成功恢复图像: {image_path} -> {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复图像失败 {image_path}: {e}")
            return False
    
    def restore_batch(self, input_dir: str, output_dir: str) -> dict:
        """
        批量恢复图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            dict: 处理结果统计
        """
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 获取所有图像文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [
            f for f in input_path.iterdir() 
            if f.suffix.lower() in image_extensions
        ]
        
        if not image_files:
            self.logger.warning(f"在 {input_dir} 中未找到图像文件")
            return {"total": 0, "success": 0, "failed": 0}
        
        self.logger.info(f"开始批量处理 {len(image_files)} 张图像")
        
        results = {"total": len(image_files), "success": 0, "failed": 0}
        
        for image_file in image_files:
            output_file = output_path / f"restored_{image_file.name}"
            
            if self.restore_single_image(str(image_file), str(output_file)):
                results["success"] += 1
            else:
                results["failed"] += 1
        
        self.logger.info(f"批量处理完成: {results}")
        return results
    
    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        """图像预处理"""
        # 转换为RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 调整大小到模型输入尺寸
        image = cv2.resize(image, (256, 256))
        
        # 归一化到 [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # 转换为tensor并添加batch维度
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def _postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """图像后处理"""
        # 移除batch维度并转换回numpy
        image = tensor.squeeze(0).cpu().numpy()
        
        # 调整通道顺序
        image = np.transpose(image, (1, 2, 0))
        
        # 裁剪到 [0, 1] 范围
        image = np.clip(image, 0, 1)
        
        # 转换到 [0, 255] 范围
        image = (image * 255).astype(np.uint8)
        
        return image


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="DefakeHop 人脸恢复")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--model_path", help="预训练模型路径")
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
    
    # 创建恢复器
    restorer = DefakeHopRestorer(model_path=args.model_path, device=args.device)
    
    # 执行恢复
    results = restorer.restore_batch(args.input_dir, args.output_dir)
    
    # 输出结果
    logger.info(f"处理完成: 总计 {results['total']} 张，成功 {results['success']} 张，失败 {results['failed']} 张")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
