# ETMA: EfficientNet-Transformer with Multi-Attention Aggregation

## Project Overview

ETMA (EfficientNet-Transformer with Multi-Attention Aggregation) is a deep learning model designed for **Frame-wise Heterogeneous Deepfake Detection** tasks. The model combines EfficientNet feature extraction, multi-attention mechanisms, Transformer sequence modeling, and prompt learning technologies to effectively identify various deepfake techniques in videos, including DeepFake, Face2Face, FaceSwap, NeuralTextures, etc.

## Key Features

- **Multi-Attention Aggregation**: Captures different levels of forgery features through multiple attention heads, achieving effective frame-level feature aggregation
- **Texture Enhancement Mechanism**: Specially designed texture enhancement module optimized for inconsistencies in texture levels of deepfakes
- **Prompt Learning**: Learnable prompt generator that generates specific prompts for different forgery types, enhancing feature representation capability
- **Dual-Level Prediction**: Supports both frame-level and video-level prediction, providing fine-grained and overall judgment
- **Deepfake Recovery**: Integrates multiple recovery algorithms including DefakeHop, GAN recovery, neural texture recovery, etc., supporting inverse recovery of detected forged content

## Dataset

### FH-DeepFake-v1 Dataset

FH-DeepFake-v1 is a benchmark dataset designed for **Frame-wise Heterogeneous Deepfake Detection** tasks. This dataset introduces **multi-label frame-level annotation** for the first time, supporting mixed multiple forgery methods within the same video.

#### Dataset Characteristics
- **Frame-level Heterogeneous Annotation**: Each frame can have multiple overlapping forgery labels (e.g., "DeepFake+FaceSwap"), covering composite tampering in real scenarios
- **Diverse Forgery Techniques**: Generated based on 4 mainstream methods: DeepFake, Face2Face, FaceSwap, NeuralTextures
- **Large-scale Data**: 15,000 videos (1,000 source videos × 15 forgery combinations), totaling over 3 million frames

#### Access Method
**Please send an email to the author's email for download link:**  
📧 **1289741281@qq.com**  
Email subject format: `[FH-DeepFake-v1 Dataset Application] Institution-Name`  
Email body should include: ① Research institution/school name ② Brief description of research purpose ③ Commitment to comply with academic use agreement

## Evaluation Metrics

- **Video-level Metrics**: EACC, PACC, MACC, Precision, Recall, F1-Score, AUC-ROC
- **Frame-level Metrics**: Frame-level detection accuracy, frame-level confidence analysis, detected frame count statistics

## Usage License

- **Academic research use only**, commercial use prohibited
- Cite this work and acknowledge in derivative works

## Contact

- Email: 1289741281@qq.com


**Note**: This project is for academic research purposes only. Please comply with relevant laws, regulations, and ethical guidelines.
