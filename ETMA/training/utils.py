import torch

def save_model(model, path):
    torch.save(model.state_dict(), path)

def adjust_lr(optimizer, epoch, initial_lr, lr_decay_epoch=10, decay_rate=0.1):
    lr = initial_lr * (decay_rate ** (epoch // lr_decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr 

def get_prompt_parameters(model):
    """Safely get prompt-related parameters"""
    prompt_params = []
    
    if hasattr(model, 'prompt_learner'):
        # Get ctx parameters
        if hasattr(model.prompt_learner, 'ctx'):
            prompt_params.append(model.prompt_learner.ctx)
        
        # Get projection parameters
        if hasattr(model.prompt_learner, 'projection'):
            if hasattr(model.prompt_learner.projection, 'weight'):
                prompt_params.append(model.prompt_learner.projection.weight)
            if hasattr(model.prompt_learner.projection, 'bias'):
                prompt_params.append(model.prompt_learner.projection.bias)
    
    return prompt_params

def get_base_parameters(model):
    """Get base model parameters (excluding prompt parameters)"""
    base_params = []
    for name, param in model.named_parameters():
        if 'prompt_learner' not in name:
            base_params.append(param)
    return base_params

def setup_prompt_optimizer(model, lr=1e-4):
    """Setup optimizer for prompt model"""
    if hasattr(model, 'prompt_learner'):
        # Freeze CLIP text encoder
        for name, param in model.prompt_learner.named_parameters():
            if 'text_encoder' in name:
                param.requires_grad = False
        
        # Get parameters
        prompt_params = get_prompt_parameters(model)
        base_params = get_base_parameters(model)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(prompt_params + base_params, lr=lr)
        
        # Print parameter statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in prompt_params + base_params)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Frozen parameter ratio: {(1 - trainable_params/total_params)*100:.2f}%")
        
        return optimizer
    else:
        return torch.optim.Adam(model.parameters(), lr=lr) 