import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, root_dir, label_names, split='train', transform=None, frames_per_video=16):
        self.root_dir = root_dir
        self.label_names = label_names  # List of label names
        self.split = split  # Dataset split: 'train', 'val', 'test'
        self.transform = transform
        self.frames_per_video = frames_per_video
        self.samples = self._gather_samples()

    def _gather_samples(self):
        # Traverse label folders in specified split
        split_dir = os.path.join(self.root_dir, self.split)
        video_dict = {}
        for idx, label in enumerate(self.label_names):
            label_dir = os.path.join(split_dir, label)
            if not os.path.isdir(label_dir):
                continue
            for video in os.listdir(label_dir):
                if video.startswith('.'):
                    continue  # Skip hidden folders
                video_path = os.path.join(label_dir, video)
                if not os.path.isdir(video_path):
                    continue
                if video not in video_dict:
                    video_dict[video] = {'labels': [0] * len(self.label_names), 'paths': []}
                video_dict[video]['labels'][idx] = 1
                video_dict[video]['paths'].append(video_path)
        samples = []
        for video, info in video_dict.items():
            samples.append((info['paths'][0], info['labels']))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, labels = self.samples[idx]
        video_name = os.path.basename(video_path)
        frame_files = sorted([f for f in os.listdir(video_path) if f.lower().endswith(('jpg', 'jpeg', 'png'))])
        if len(frame_files) == 0:
            print(f"[DEBUG] No frames found in {video_path}")
        if len(frame_files) >= self.frames_per_video:
            indices = torch.linspace(0, len(frame_files)-1, self.frames_per_video).long()
            frame_files = [frame_files[i] for i in indices]
        images = []
        for frame in frame_files:
            img = Image.open(os.path.join(video_path, frame)).convert('RGB')
            if self.transform:
                img = self.transform(img)
            images.append(img)
        images = torch.stack(images)  # (frames, C, H, W)
        return images, torch.tensor(labels, dtype=torch.float32), video_name

def load_data(root_dir, label_names, split='train', transform=None, frames_per_video=16):
    return VideoDataset(root_dir, label_names, split, transform, frames_per_video)

def get_label_names(root_dir, split='train'):
    # Get all label names in specified split, skip hidden folders
    split_dir = os.path.join(root_dir, split)
    return sorted([d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d)) and not d.startswith('.')]) 