import torch

def train_epoch(model, train_loader, optimizer, criterion, device, epoch=None):
    model.train()
    running_loss = 0.0
    
    for images, labels, _ in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Check if UCF model, use custom loss function
        if hasattr(model, 'compute_loss') and callable(getattr(model, 'compute_loss', None)):
            loss, loss_dict = model.compute_loss(images, labels, epoch)
        else:
            # Standard forward pass
            if epoch is not None and hasattr(model, 'forward') and 'epoch' in model.forward.__code__.co_varnames:
                outputs = model(images, epoch=epoch)
            else:
                outputs = model(images)
            loss = criterion(outputs, labels)
            loss_dict = {'total_loss': loss.item()}
        
        loss.backward()
        optimizer.step()
        running_loss += loss_dict['total_loss']
    
    return running_loss / len(train_loader)

def validate(model, val_loader, criterion, device, epoch=None):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in val_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Check if UCF model, use custom loss function
            if hasattr(model, 'compute_loss') and callable(getattr(model, 'compute_loss', None)):
                loss, loss_dict = model.compute_loss(images, labels, epoch)
                # For UCF model, use fusion prediction results for accuracy calculation
                outputs = model(images)
            else:
                # Standard forward pass
                if epoch is not None and hasattr(model, 'forward') and 'epoch' in model.forward.__code__.co_varnames:
                    outputs = model(images, epoch=epoch)
                else:
                    outputs = model(images)
                loss = criterion(outputs, labels)
                loss_dict = {'total_loss': loss.item()}
            
            val_loss += loss_dict['total_loss']
            preds = (outputs > 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.numel()
    
    acc = correct / total if total > 0 else 0
    return val_loss / len(val_loader), acc 