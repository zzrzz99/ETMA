#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像后处理脚本
"""

import os
import sys
import logging
import argparse
import numpy as np
import cv2
from pathlib import Path
from PIL import Image, ImageEnhance

# 添加模块路径
sys.path.append(str(Path(__file__).parent.parent))

from .utils import setup_logging, load_image, save_image, create_output_dir, get_image_files

class ImagePostProcessor:
    """图像后处理器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def enhance_quality(self, image: np.ndarray) -> np.ndarray:
        """增强图像质量"""
        pil_image = Image.fromarray(image)
        
        # 亮度调整
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        # 对比度调整
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(1.1)
        
        return np.array(pil_image)
    
    def denoise(self, image: np.ndarray) -> np.ndarray:
        """图像去噪"""
        return cv2.bilateralFilter(image, 9, 10, 10)
    
    def postprocess_single_image(self, image_path: str, output_path: str) -> bool:
        try:
            image = load_image(image_path)
            if image is None:
                return False
            
            # 质量增强
            image = self.enhance_quality(image)
            
            # 去噪
            image = self.denoise(image)
            
            return save_image(image, output_path)
            
        except Exception as e:
            self.logger.error(f"后处理失败: {e}")
            return False
    
    def postprocess_batch(self, input_dir: str, output_dir: str):
        if not create_output_dir(output_dir):
            return
        
        image_files = get_image_files(input_dir)
        if not image_files:
            return
        
        for image_file in image_files:
            rel_path = os.path.relpath(image_file, input_dir)
            output_path = os.path.join(output_dir, rel_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            self.postprocess_single_image(image_file, output_path)

def main():
    parser = argparse.ArgumentParser(description="图像后处理")
    parser.add_argument("--input_dir", type=str, required=True, help="输入图像目录")
    parser.add_argument("--output_dir", type=str, required=True, help="输出目录")
    
    args = parser.parse_args()
    
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("启动图像后处理")
    
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return
    
    processor = ImagePostProcessor()
    processor.postprocess_batch(args.input_dir, args.output_dir)
    
    logger.info("图像后处理完成")

if __name__ == "__main__":
    main()
