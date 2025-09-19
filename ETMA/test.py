import argparse
import torch
from model.multi_label_model import MultiLabelModel
from data_preprocessing.dataset_loader import load_data, get_label_names
from data_preprocessing.transforms import get_transforms
from test.evaluator import evaluate_with_frame_predictions

# Default parameters
DEFAULT_DATA_DIR = 'data'
DEFAULT_CHECKPOINT_PATH = 'best_model.pth'
DEFAULT_FRAMES_PER_VIDEO = 16
DEFAULT_BATCH_SIZE = 2
DEFAULT_ACC_TYPE = 'all'  # Options: 'all', 'per_label', 'mean'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-label deepfake detection model testing script')
    parser.add_argument('--data_dir', type=str, default=DEFAULT_DATA_DIR, help='Test dataset root directory')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT_PATH, help='Model weight file path')
    parser.add_argument('--frames_per_video', type=int, default=DEFAULT_FRAMES_PER_VIDEO, help='Number of frames per video')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Test batch size')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    parser.add_argument('--acc_type', type=str, default=DEFAULT_ACC_TYPE, choices=['all', 'per_label', 'mean'], help='Accuracy display mode')
    args = parser.parse_args()

    # Get labels (based on test set)
    label_names = get_label_names(args.data_dir, split='test')
    print('Label classes:', label_names)

    # Load test dataset
    test_dataset = load_data(
        args.data_dir,
        label_names,
        split='test',
        transform=get_transforms('test'),
        frames_per_video=args.frames_per_video
    )
    def custom_collate(batch):
        images, labels, video_names = zip(*batch)
        return torch.stack(images), torch.stack(labels), list(video_names)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=custom_collate)

    # Initialize model
    model = MultiLabelModel(num_classes=len(label_names)).to(args.device)
    # Load weights
    model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    print(f'Loaded model weights: {args.checkpoint}')

    # Execute evaluation
    results = evaluate_with_frame_predictions(
        model,
        test_loader,
        args.device,
        label_names,
        threshold=0.5
    )
    metrics = results['video_level_metrics']
    if args.acc_type == 'all':
        print('Video-level test all-label accuracy:', metrics['accuracy'])
    elif args.acc_type == 'per_label':
        print('Video-level test per-label accuracy:', metrics['per_label_accuracy'])
    elif args.acc_type == 'mean':
        print('Video-level test mean accuracy:', metrics['mean_accuracy'])
    print('Video-level test other metrics:', {k: v for k, v in metrics.items() if k not in ['accuracy', 'per_label_accuracy', 'mean_accuracy']})
    print('Testing completed, detected {} videos'.format(len(test_dataset)))
    print('Detailed frame-level prediction results saved to frame_predictions_results.json') 