"""
人脸恢复模块 - 支持多种伪造方式的恢复方法

包含以下恢复方法：
1. DefakeHop - DeepFake 恢复
2. Neural Texture Reversion - NeuralTextures 恢复  
3. Reverting Face2Face - Face2Face 表情迁移恢复
4. DeepFaceLab-Restore + GAN Inversion - FaceSwap 恢复

作者: AI Assistant
版本: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "AI Assistant"

from . import defakehop
from . import neural_texture_reversion
from . import reverting_face2face
from . import df_restore_gan
from . import shared_utils

__all__ = [
    "defakehop",
    "neural_texture_reversion", 
    "reverting_face2face",
    "df_restore_gan",
    "shared_utils"
]
