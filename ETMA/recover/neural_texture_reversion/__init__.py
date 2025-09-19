"""
Neural Texture Reversion - NeuralTextures 人脸恢复模块

专门针对NeuralTextures纹理伪造进行恢复的方法，基于纹理分析和重建技术。

主要功能：
- 纹理伪造检测
- 纹理恢复重建
- 质量评估
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .restore import NeuralTextureRestorer
from .model import NeuralTextureModel

__all__ = ["NeuralTextureRestorer", "NeuralTextureModel"]
