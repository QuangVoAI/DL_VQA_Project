"""
Training Module for VQA Model

This module contains the training loop and related functions for training
the Visual Question Answering model.
"""

import os
import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import VQADataset, Vocabulary
from model import VQAModel


def train_epoch(model: nn.Module,
                dataloader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int) -> float:
    """
    Train model for one epoch.
    
    Args:
        model (nn.Module): VQA model
        dataloader (DataLoader): Training data loader
        criterion (nn.Module): Loss function
        optimizer (optim.Optimizer): Optimizer
        device (torch.device): Device to train on
        epoch (int): Current epoch number
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch_idx, (images, questions, answers, metadata) in enumerate(pbar):
        # Move data to device
        images = images.to(device)
        questions = questions.to(device)
        answers = answers.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        # Output shape: (batch_size, question_length, vocab_size)
        outputs = model(images, questions)
        
        # Reshape for loss calculation
        # outputs: (batch_size * sequence_length, vocab_size)
        # answers: (batch_size * sequence_length)
        batch_size, seq_length, vocab_size = outputs.shape
        outputs_flat = outputs.reshape(-1, vocab_size)
        answers_flat = answers.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs_flat, answers_flat)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        
        # Update weights
        optimizer.step()
        
        # Accumulate loss
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def validate_epoch(model: nn.Module,
                   dataloader: DataLoader,
                   criterion: nn.Module,
                   device: torch.device,
                   epoch: int) -> float:
    """
    Validate model for one epoch.
    
    Args:
        model (nn.Module): VQA model
        dataloader (DataLoader): Validation data loader
        criterion (nn.Module): Loss function
        device (torch.device): Device to validate on
        epoch (int): Current epoch number
        
    Returns:
        float: Average validation loss for the epoch
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    # Progress bar
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    
    with torch.no_grad():
        for images, questions, answers, metadata in pbar:
            # Move data to device
            images = images.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            
            # Forward pass
            outputs = model(images, questions)
            
            # Reshape for loss calculation
            batch_size, seq_length, vocab_size = outputs.shape
            outputs_flat = outputs.reshape(-1, vocab_size)
            answers_flat = answers.reshape(-1)
            
            # Calculate loss
            loss = criterion(outputs_flat, answers_flat)
            
            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    return avg_loss


def save_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   epoch: int,
                   loss: float,
                   checkpoint_dir: str,
                   filename: Optional[str] = None):
    """
    Save model checkpoint.
    
    Args:
        model (nn.Module): Model to save
        optimizer (optim.Optimizer): Optimizer state to save
        epoch (int): Current epoch
        loss (float): Current loss
        checkpoint_dir (str): Directory to save checkpoint
        filename (Optional[str]): Checkpoint filename (default: checkpoint_epoch_{epoch}.pth)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if filename is None:
        filename = f'checkpoint_epoch_{epoch}.pth'
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f'Checkpoint saved: {checkpoint_path}')


def load_checkpoint(model: nn.Module,
                   optimizer: optim.Optimizer,
                   checkpoint_path: str,
                   device: torch.device) -> int:
    """
    Load model checkpoint.
    
    Args:
        model (nn.Module): Model to load weights into
        optimizer (optim.Optimizer): Optimizer to load state into
        checkpoint_path (str): Path to checkpoint file
        device (torch.device): Device to load checkpoint on
        
    Returns:
        int: Epoch number from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f'Checkpoint loaded: {checkpoint_path}')
    print(f'Resuming from epoch {epoch} with loss {loss:.4f}')
    
    return epoch


def train(args):
    """
    Main training function.
    
    Args:
        args: Command-line arguments
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Image transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load or build vocabulary
    vocab_path = os.path.join(args.data_path, 'vocab.json')
    if os.path.exists(vocab_path):
        print(f'Loading vocabulary from {vocab_path}')
        vocab = Vocabulary.load(vocab_path)
    else:
        print('Building vocabulary...')
        # TODO: Build vocabulary from training data
        vocab = Vocabulary()
        # vocab.build_vocabulary(training_sentences)
        vocab.save(vocab_path)
        print(f'Vocabulary saved to {vocab_path}')
    
    print(f'Vocabulary size: {len(vocab)}')
    
    # Create datasets
    # TODO: Update paths based on your dataset structure
    train_dataset = VQADataset(
        image_dir=os.path.join(args.data_path, 'images/train'),
        questions_file=os.path.join(args.data_path, 'questions_train.json'),
        answers_file=os.path.join(args.data_path, 'answers_train.json'),
        vocab=vocab,
        transform=train_transforms,
        max_question_length=args.max_question_length,
        max_answer_length=args.max_answer_length
    )
    
    val_dataset = VQADataset(
        image_dir=os.path.join(args.data_path, 'images/val'),
        questions_file=os.path.join(args.data_path, 'questions_val.json'),
        answers_file=os.path.join(args.data_path, 'answers_val.json'),
        vocab=vocab,
        transform=val_transforms,
        max_question_length=args.max_question_length,
        max_answer_length=args.max_answer_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    print(f'Training samples: {len(train_dataset)}')
    print(f'Validation samples: {len(val_dataset)}')
    
    # Create model
    model = VQAModel(
        vocab_size=len(vocab),
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        image_feature_dim=args.image_feature_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        pretrained_encoder=True,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    print(f'Model parameters: {sum(p.numel() for p in model.parameters()):,}')
    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    # Loss function (ignore padding token)
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.word2idx[vocab.pad_token])
    
    # Optimizer
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )
    
    # Load checkpoint if specified
    start_epoch = 1
    if args.resume_checkpoint:
        start_epoch = load_checkpoint(model, optimizer, args.resume_checkpoint, device) + 1
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.num_epochs + 1):
        print(f'\n{"="*50}')
        print(f'Epoch {epoch}/{args.num_epochs}')
        print(f'{"="*50}')
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        print(f'Train Loss: {train_loss:.4f}')
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, epoch)
        print(f'Validation Loss: {val_loss:.4f}')
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save checkpoint
        save_checkpoint(model, optimizer, epoch, val_loss, args.checkpoint_dir)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, epoch, val_loss, 
                args.checkpoint_dir, filename='best_model.pth'
            )
            print(f'New best model saved! Val Loss: {val_loss:.4f}')
    
    print('\nTraining completed!')


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train VQA Model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, default='data/',
                       help='Path to dataset directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                       help='Directory to save checkpoints')
    parser.add_argument('--max_question_length', type=int, default=20,
                       help='Maximum question length')
    parser.add_argument('--max_answer_length', type=int, default=10,
                       help='Maximum answer length')
    
    # Model parameters
    parser.add_argument('--embed_dim', type=int, default=256,
                       help='Word embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                       help='LSTM hidden dimension')
    parser.add_argument('--image_feature_dim', type=int, default=2048,
                       help='Image feature dimension')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                       help='Dropout probability')
    parser.add_argument('--freeze_encoder', action='store_true',
                       help='Freeze encoder weights')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                       help='Path to checkpoint to resume training from')
    
    args = parser.parse_args()
    
    # Start training
    train(args)


if __name__ == '__main__':
    main()
