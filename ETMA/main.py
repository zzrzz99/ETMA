import os
import torch
import json
from torch.utils.data import DataLoader
from data_preprocessing.dataset_loader import load_data, get_label_names
from data_preprocessing.transforms import get_transforms
from model.multi_label_model import MultiLabelModel
from training.trainer import train_epoch, validate
from training.utils import save_model, adjust_lr, setup_prompt_optimizer
from test.evaluator import evaluate, evaluate_with_frame_predictions

# Configuration parameters
DATA_DIR = '/root/autodl-tmp/datasets'  # Dataset root directory
FRAMES_PER_VIDEO = 16
BATCH_SIZE = 2
EPOCHS = 20
LR = 1e-4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
RESULT_JSON = 'train_test_results.json'
ACC_TYPE = 'all'  # Options: 'all', 'per_label', 'mean'

if __name__ == '__main__':
    # 1. Get label names automatically
    label_names = get_label_names(DATA_DIR, split='train')
    NUM_CLASSES = len(label_names)
    print('Label classes:', label_names)

    # 2. Load train/val/test sets
    train_dataset = load_data(DATA_DIR, label_names, split='train', transform=get_transforms('train'), frames_per_video=FRAMES_PER_VIDEO)
    val_dataset = load_data(DATA_DIR, label_names, split='val', transform=get_transforms('val'), frames_per_video=FRAMES_PER_VIDEO)
    test_dataset = load_data(DATA_DIR, label_names, split='test', transform=get_transforms('test'), frames_per_video=FRAMES_PER_VIDEO)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    # 3. Build model
    model = MultiLabelModel(num_classes=NUM_CLASSES, use_prompt=True).to(DEVICE)
    criterion = torch.nn.BCELoss()
    
    # Setup optimizer
    optimizer = setup_prompt_optimizer(model, lr=LR)

    # 4. Training
    history = []
    for epoch in range(EPOCHS):
        loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        adjust_lr(optimizer, epoch, LR)
        print(f'Epoch {epoch+1}/{EPOCHS} | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
        save_model(model, f'checkpoint_epoch{epoch+1}.pth')
        history.append({
            'epoch': epoch+1,
            'train_loss': loss,
            'val_loss': val_loss,
            'val_acc': val_acc
        })

    # 5. Testing (with frame-level predictions)
    print("Starting testing and generating frame-level prediction results...")
    detailed_results = evaluate_with_frame_predictions(model, test_loader, DEVICE, label_names)
    metrics = detailed_results['video_level_metrics']
    if ACC_TYPE == 'all':
        print('Video-level test all-label accuracy:', metrics['accuracy'])
    elif ACC_TYPE == 'per_label':
        print('Video-level test per-label accuracy:', metrics['per_label_accuracy'])
    elif ACC_TYPE == 'mean':
        print('Video-level test mean accuracy:', metrics['mean_accuracy'])
    print('Video-level test other metrics:', {k: v for k, v in metrics.items() if k not in ['accuracy', 'per_label_accuracy', 'mean_accuracy']})

    # 6. Save all results to json
    result = {
        'train_history': history,
        'test_metrics': metrics,
        'frame_predictions': detailed_results['frame_level_predictions']
    }
    with open(RESULT_JSON, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f'Training and testing results saved to {RESULT_JSON}')
    print(f'Frame-level prediction results saved to frame_predictions_results.json') 