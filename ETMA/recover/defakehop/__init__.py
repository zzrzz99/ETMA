"""
DefakeHop - DeepFake 人脸恢复模块

基于深度学习的DeepFake检测和恢复方法，专门针对换脸伪造进行恢复。

主要功能：
- DeepFake检测
- 人脸恢复
- 质量评估
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .restore import DefakeHopRestorer
from .model import DefakeHopModel

__all__ = ["DefakeHopRestorer", "DefakeHopModel"]
