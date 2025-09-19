#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
General utility functions module

Provides common functions for image processing, logging management, file operations, etc.
"""

import os
import logging
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import Optional, Union


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[]
    )
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(console_handler)
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    logging.info(f"日志系统初始化完成，级别: {level}")


def load_image(image_path: str) -> Optional[np.ndarray]:
    """加载图像文件"""
    try:
        image = cv2.imread(image_path)
        if image is not None:
            return image
        
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        return image
        
    except Exception as e:
        logging.error(f"加载图像失败 {image_path}: {e}")
        return None


def save_image(image: np.ndarray, output_path: str, quality: int = 95) -> bool:
    """保存图像文件"""
    try:
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        cv2.imwrite(output_path, image)
        return True
        
    except Exception as e:
        logging.error(f"保存图像失败 {output_path}: {e}")
        return False


def create_output_dir(output_path: str) -> bool:
    """创建输出目录"""
    try:
        os.makedirs(output_path, exist_ok=True)
        return True
    except Exception as e:
        logging.error(f"创建目录失败 {output_path}: {e}")
        return False


def get_image_files(input_dir: str, extensions: Optional[set] = None) -> list:
    """获取目录中的图像文件"""
    if extensions is None:
        extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    image_files = []
    input_path = Path(input_dir)
    
    if not input_path.exists():
        logging.warning(f"目录不存在: {input_dir}")
        return image_files
    
    for file_path in input_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in extensions:
            image_files.append(str(file_path))
    
    return sorted(image_files)
