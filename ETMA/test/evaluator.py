import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json


def evaluate(model, test_loader, device):
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = outputs.cpu().numpy()
            preds = (outputs > 0.5).cpu().numpy()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='samples', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='samples', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='samples', zero_division=0)
    # Per-label accuracy and mean accuracy
    per_label_acc = (all_labels == all_preds).float().mean(dim=0).tolist()
    mean_acc = (all_labels == all_preds).float().mean().item()
    try:
        auc = roc_auc_score(all_labels, all_probs, average='macro')
    except Exception:
        auc = None
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1,
            'per_label_accuracy': per_label_acc, 'mean_accuracy': mean_acc, 'auc': auc}


def evaluate_with_frame_predictions(model, test_loader, device, label_names, threshold=0.5, epoch=None):
    """
    Evaluate model and output frame-level prediction results
    """
    model.eval()
    all_video_labels = []
    all_video_preds = []
    all_video_probs = []
    frame_predictions_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            if len(batch) == 3:
                images, labels, video_names = batch
            else:
                images, labels = batch
                video_names = [f"video_{batch_idx}_{i}" for i in range(images.size(0))]
            images = images.to(device)
            if epoch is not None and hasattr(model, 'forward') and 'epoch' in model.forward.__code__.co_varnames:
                video_probs, frame_preds = model(images, return_frame_predictions=True, epoch=epoch)
            else:
                video_probs, frame_preds = model(images, return_frame_predictions=True)

            # Video-level prediction
            video_probs_np = video_probs.cpu().numpy()
            video_preds = (video_probs > threshold).cpu().numpy()
            all_video_probs.extend(video_probs_np)
            all_video_preds.extend(video_preds)
            all_video_labels.extend(labels.numpy())

            # Frame-level prediction processing
            frame_preds = frame_preds.cpu().numpy()  # (batch, frames, num_classes)
            batch_size = images.size(0)
            for i in range(batch_size):  # Iterate through each video
                video_result = {
                    'video_id': video_names[i],
                    'video_level_predictions': {},
                    'frame_level_predictions': {}
                }

                # Video-level prediction results
                current_video_idx = len(all_video_preds) - batch_size + i  # Calculate current video index
                for j, label_name in enumerate(label_names):
                    video_result['video_level_predictions'][label_name] = {
                        'predicted': bool(video_preds[i][j]),
                        'confidence': float(video_probs_np[i][j])
                    }

                # Frame-level prediction results
                for j, label_name in enumerate(label_names):
                    frame_indices = []
                    frame_confidences = []

                    for frame_idx in range(frame_preds.shape[1]):  # Iterate through each frame
                        if frame_preds[i, frame_idx, j] > threshold:
                            frame_indices.append(int(frame_idx))  # Ensure int type
                            frame_confidences.append(float(frame_preds[i, frame_idx, j]))

                    video_result['frame_level_predictions'][label_name] = {
                        'detected_frames': frame_indices,
                        'frame_confidences': frame_confidences,
                        'total_detected_frames': len(frame_indices)
                    }

                frame_predictions_results.append(video_result)

    # Calculate video-level evaluation metrics
    all_video_preds = torch.tensor(all_video_preds)
    all_video_labels = torch.tensor(all_video_labels)
    all_video_probs = torch.tensor(all_video_probs)
    acc = accuracy_score(all_video_labels, all_video_preds)
    precision = precision_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    recall = recall_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    f1 = f1_score(all_video_labels, all_video_preds, average='samples', zero_division=0)
    
    # Fix data type issues: ensure all data is JSON serializable
    per_label_acc = (all_video_labels == all_video_preds).float().mean(dim=0)
    per_label_acc_list = [float(acc.item()) for acc in per_label_acc]  # Convert to Python float list
    mean_acc = float((all_video_labels == all_video_preds).float().mean().item())
    
    try:
        auc = roc_auc_score(all_video_labels, all_video_probs, average='macro')
        auc = float(auc) if auc is not None else None
    except Exception:
        auc = None
    
    # Save detailed results to JSON
    detailed_results = {
        'video_level_metrics': {
            'accuracy': float(acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1),
            'per_label_accuracy': per_label_acc_list,
            'mean_accuracy': mean_acc,
            'auc': auc
        },
        'frame_level_predictions': frame_predictions_results
    }

    with open('frame_predictions_results.json', 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)

    return detailed_results
