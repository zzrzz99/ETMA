#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face detection and cropping module

High-precision face detection and cropping using RetinaFace
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
import logging


class FaceDetector:
    """Face detector"""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize face detector
        
        Args:
            model_path: RetinaFace model path, use default model if None
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize RetinaFace detector
        try:
            if model_path:
                self.detector = cv2.dnn.readNetFromCaffe(
                    model_path + "/deploy.prototxt",
                    model_path + "/res10_300x300_ssd_iter_140000.caffemodel"
                )
            else:
                # 使用OpenCV内置的人脸检测器作为备选
                self.detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
            self.logger.info("人脸检测器初始化成功")
        except Exception as e:
            self.logger.warning(f"RetinaFace初始化失败: {e}，使用OpenCV内置检测器")
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
    
    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        检测图像中的人脸
        
        Args:
            image: 输入图像
            
        Returns:
            List[Tuple[int, int, int, int]]: 人脸边界框列表 (x, y, w, h)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        try:
            # 尝试使用RetinaFace
            if hasattr(self.detector, 'forward'):
                # DNN模型
                blob = cv2.dnn.blobFromImage(
                    cv2.resize(image, (300, 300)), 
                    1.0, (300, 300), (104.0, 177.0, 123.0)
                )
                self.detector.setInput(blob)
                detections = self.detector.forward()
                
                faces = []
                height, width = image.shape[:2]
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
                        x, y, x2, y2 = box.astype(int)
                        faces.append((x, y, x2 - x, y2 - y))
                        
                return faces
            else:
                # OpenCV级联分类器
                faces = self.detector.detectMultiScale(
                    gray, 
                    scaleFactor=1.1, 
                    minNeighbors=5, 
                    minSize=(30, 30)
                )
                return [(x, y, w, h) for (x, y, w, h) in faces]
                
        except Exception as e:
            self.logger.error(f"人脸检测失败: {e}")
            return []
    
    def crop_face(self, image: np.ndarray, face_box: Tuple[int, int, int, int], 
                  target_size: int = 512, margin: float = 0.2) -> np.ndarray:
        """
        裁剪人脸区域
        
        Args:
            image: 输入图像
            face_box: 人脸边界框 (x, y, w, h)
            target_size: 目标尺寸
            margin: 边界扩展比例
            
        Returns:
            np.ndarray: 裁剪后的人脸图像
        """
        x, y, w, h = face_box
        
        # 计算扩展边界
        margin_x = int(w * margin)
        margin_y = int(h * margin)
        
        # 扩展边界框
        x1 = max(0, x - margin_x)
        y1 = max(0, y - margin_y)
        x2 = min(image.shape[1], x + w + margin_x)
        y2 = min(image.shape[0], y + h + margin_y)
        
        # 裁剪人脸
        face_crop = image[y1:y2, x1:x2]
        
        # 调整到目标尺寸
        face_resized = cv2.resize(face_crop, (target_size, target_size))
        
        return face_resized
    
    def process_image(self, image: np.ndarray, target_size: int = 512, 
                     margin: float = 0.2) -> List[np.ndarray]:
        """
        处理单张图像，检测并裁剪所有人脸
        
        Args:
            image: 输入图像
            target_size: 目标尺寸
            margin: 边界扩展比例
            
        Returns:
            List[np.ndarray]: 裁剪后的人脸图像列表
        """
        # 检测人脸
        faces = self.detect_faces(image)
        
        if not faces:
            self.logger.warning("未检测到人脸")
            return []
        
        # 裁剪人脸
        cropped_faces = []
        for face_box in faces:
            face_crop = self.crop_face(image, face_box, target_size, margin)
            cropped_faces.append(face_crop)
        
        self.logger.info(f"检测到 {len(faces)} 个人脸")
        return cropped_faces


def create_face_detector(model_path: Optional[str] = None) -> FaceDetector:
    """
    创建人脸检测器实例
    
    Args:
        model_path: 模型路径
        
    Returns:
        FaceDetector: 检测器实例
    """
    return FaceDetector(model_path)


if __name__ == "__main__":
    # 测试人脸检测器
    detector = create_face_detector()
    
    # 读取测试图像
    test_image = cv2.imread("test_image.jpg")
    if test_image is not None:
        # 检测人脸
        faces = detector.detect_faces(test_image)
        print(f"检测到 {len(faces)} 个人脸")
        
        # 裁剪人脸
        for i, face_box in enumerate(faces):
            face_crop = detector.crop_face(test_image, face_box)
            cv2.imwrite(f"face_{i}.jpg", face_crop)
    else:
        print("请提供测试图像")
