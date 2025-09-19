"""
DeepFaceLab-Restore + GAN Inversion - FaceSwap 恢复模块

结合DeepFaceLab恢复技术和GAN反演的人脸恢复方法，专门针对FaceSwap换脸伪造。

主要功能：
- FaceSwap检测
- GAN反演恢复
- 质量评估
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .restore import FaceSwapRestorer
from .model import FaceSwapModel

__all__ = ["FaceSwapRestorer", "FaceSwapModel"]
