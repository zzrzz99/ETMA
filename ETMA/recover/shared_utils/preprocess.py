#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图像预处理脚本

批量处理图像，进行人脸检测、裁剪和预处理
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from typing import List

import cv2
import numpy as np

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from .face_detection import FaceDetector
from .utils import setup_logging, load_image, save_image, create_output_dir, get_image_files


def preprocess_images(input_dir: str, output_dir: str, face_size: int = 512, 
                     margin: float = 0.2, model_path: str = None) -> dict:
    """
    批量预处理图像
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        face_size: 人脸尺寸
        margin: 边界扩展比例
        model_path: 人脸检测模型路径
        
    Returns:
        dict: 处理结果统计
    """
    # 创建输出目录
    if not create_output_dir(output_dir):
        return {"total": 0, "success": 0, "failed": 0}
    
    # 获取图像文件列表
    image_files = get_image_files(input_dir)
    
    if not image_files:
        logging.warning(f"在 {input_dir} 中未找到图像文件")
        return {"total": 0, "success": 0, "failed": 0}
    
    # 创建人脸检测器
    detector = FaceDetector(model_path)
    
    logging.info(f"开始预处理 {len(image_files)} 张图像")
    
    results = {"total": len(image_files), "success": 0, "failed": 0}
    
    for image_file in image_files:
        try:
            # 加载图像
            image = load_image(image_file)
            if image is None:
                results["failed"] += 1
                continue
            
            # 检测人脸
            faces = detector.detect_faces(image)
            
            if not faces:
                logging.warning(f"在 {image_file} 中未检测到人脸")
                results["failed"] += 1
                continue
            
            # 处理每个人脸
            for i, face_box in enumerate(faces):
                # 裁剪人脸
                face_crop = detector.crop_face(image, face_box, face_size, margin)
                
                # 生成输出文件名
                base_name = Path(image_file).stem
                if len(faces) > 1:
                    output_name = f"{base_name}_face_{i}.jpg"
                else:
                    output_name = f"{base_name}.jpg"
                
                output_path = os.path.join(output_dir, output_name)
                
                # 保存裁剪后的人脸
                if save_image(face_crop, output_path):
                    results["success"] += 1
                    logging.info(f"成功处理: {image_file} -> {output_path}")
                else:
                    results["failed"] += 1
                    
        except Exception as e:
            logging.error(f"处理图像失败 {image_file}: {e}")
            results["failed"] += 1
    
    logging.info(f"预处理完成: {results}")
    return results


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="图像预处理 - 人脸检测与裁剪")
    parser.add_argument("--input_dir", required=True, help="输入图像目录")
    parser.add_argument("--output_dir", required=True, help="输出目录")
    parser.add_argument("--face_size", type=int, default=512, help="人脸尺寸")
    parser.add_argument("--margin", type=float, default=0.2, help="边界扩展比例")
    parser.add_argument("--model_path", help="人脸检测模型路径")
    parser.add_argument("--log_level", default="INFO", help="日志级别")
    
    args = parser.parse_args()
    
    # 设置日志
    setup_logging(level=args.log_level)
    logger = logging.getLogger(__name__)
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        return 1
    
    # 执行预处理
    results = preprocess_images(
        args.input_dir, 
        args.output_dir, 
        args.face_size, 
        args.margin, 
        args.model_path
    )
    
    # 输出结果
    logger.info(f"处理完成: 总计 {results['total']} 张，成功 {results['success']} 张，失败 {results['failed']} 张")
    
    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
