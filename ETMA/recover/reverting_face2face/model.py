#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Face2Face 恢复模型 - 基于 3DMM 表情参数解耦与反演
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Face2FaceModel(nn.Module):
    """Face2Face 恢复模型 - 3DMM 表情参数反演"""
    
    def __init__(self, exp_dim: int = 64, id_dim: int = 80, pose_dim: int = 6):
        super(Face2FaceModel, self).__init__()
        self.exp_dim = exp_dim
        self.id_dim = id_dim
        self.pose_dim = pose_dim
        
        # 表情参数反演网络
        self.exp_inverter = nn.Sequential(
            nn.Linear(exp_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, exp_dim)
        )
        
        # 3DMM 重建网络
        self.fusion_net = nn.Sequential(
            nn.Linear(exp_dim + id_dim + pose_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 256 * 256)
        )
        
        # 渲染网络
        self.renderer = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, exp_fake: torch.Tensor, id_coeff: torch.Tensor, 
                pose_coeff: torch.Tensor) -> torch.Tensor:
        # 表情参数反演
        exp_restored = self.exp_inverter(exp_fake)
        
        # 3D 人脸重建
        combined = torch.cat([exp_restored, id_coeff, pose_coeff], dim=1)
        vertices = self.fusion_net(combined)
        face_3d = vertices.view(-1, 3, 256, 256)
        
        # 可微渲染
        restored_face = self.renderer(face_3d)
        
        return restored_face

class Face2FaceRestorer(nn.Module):
    """Face2Face 恢复器完整模型"""
    
    def __init__(self, exp_dim: int = 64, id_dim: int = 80, pose_dim: int = 6):
        super(Face2FaceRestorer, self).__init__()
        self.main_net = Face2FaceModel(exp_dim, id_dim, pose_dim)
        
    def forward(self, exp_fake: torch.Tensor, id_coeff: torch.Tensor, 
                pose_coeff: torch.Tensor) -> torch.Tensor:
        return self.main_net(exp_fake, id_coeff, pose_coeff)
