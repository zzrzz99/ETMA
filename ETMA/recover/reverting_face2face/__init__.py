"""
Reverting Face2Face - Face2Face 表情迁移恢复模块

专门针对Face2Face表情迁移伪造进行恢复的方法，基于表情分析和重建技术。

主要功能：
- 表情伪造检测
- 表情恢复重建
- 质量评估
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from .restore import Face2FaceRestorer
from .model import Face2FaceModel

__all__ = ["Face2FaceRestorer", "Face2FaceModel"]
