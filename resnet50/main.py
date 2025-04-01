import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import torch.nn as nn
import torch.optim as optim
import argparse
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def parse_train_args():
    parser = argparse.ArgumentParser(description='Train ResNet50 on Fashion MNIST')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=64)
    # Optionally, you could also add arguments for world_size, etc.
    return parser.parse_args()

def setup_logging(main_process):
    import sys
    if main_process:
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s',
                            handlers=[logging.StreamHandler(sys.stdout)])

def init_distributed():
    # Check if running in distributed mode by looking for LOCAL_RANK environment variable.
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    # Initialize the process group if not already initialized.
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return device, local_rank

def main():
    train_args = parse_train_args()
    
    # Initialize distributed training if applicable.
    device, local_rank = init_distributed()
    main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
    setup_logging(main_process)
    
    if main_process:
        logging.info(f"Training arguments: {train_args}")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = torchvision.datasets.FashionMNIST(root='.', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.FashionMNIST(root='.', train=False, download=True, transform=transform)
    
    # Create distributed samplers.
    train_sampler = DistributedSampler(train_dataset) if dist.is_initialized() else None
    test_sampler = DistributedSampler(test_dataset, shuffle=False) if dist.is_initialized() else None
    
    train_loader = DataLoader(train_dataset, batch_size=train_args.batch_size, shuffle=(train_sampler is None), sampler=train_sampler)
    test_loader = DataLoader(test_dataset, batch_size=train_args.batch_size, shuffle=False, sampler=test_sampler)
    
    model = resnet50(pretrained=True)
    # Adapt first conv layer to accept 1 channel input.
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)
    model = model.to(device)
    
    # Wrap model in DistributedDataParallel.
    if dist.is_initialized():
        model = DDP(model, device_ids=[local_rank] if torch.cuda.is_available() else None)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(train_args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
            
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            if (batch_idx + 1) % 100 == 0 and main_process:
                logging.info(f'Epoch [{epoch+1}/{train_args.epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Distributed evaluation: accumulate loss and accuracy over all processes.
        model.eval()
        test_loss_sum = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                # Multiply loss by the batch size for correct aggregation.
                test_loss_sum += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        # Reduce metrics from all processes.
        if dist.is_initialized():
            test_loss_tensor = torch.tensor(test_loss_sum, device=device)
            test_correct_tensor = torch.tensor(test_correct, device=device)
            test_total_tensor = torch.tensor(test_total, device=device)
            dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total_tensor, op=dist.ReduceOp.SUM)
            avg_test_loss = test_loss_tensor.item() / test_total_tensor.item()
            test_accuracy = 100 * test_correct_tensor.item() / test_total_tensor.item()
        else:
            avg_test_loss = test_loss_sum / test_total
            test_accuracy = 100 * test_correct / test_total
        
        if main_process:
            logging.info(f'Epoch [{epoch+1}/{train_args.epochs}]:')
            logging.info(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
            logging.info(f'  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
    
    if dist.is_initialized():
        dist.barrier()  # Synchronize all processes before final evaluation
        logging.info('Training finished, start evaluation on test set.')
    
    torch.cuda.empty_cache()

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    if dist.is_initialized():
        # Reduce metrics from all processes
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        correct = correct_tensor.item()
        total = total_tensor.item()

    if main_process:
        logging.info(f'Final Test Accuracy: {100 * correct / total:.2f}%')
    
    # Clean up the process group.
    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == '__main__':
    main()