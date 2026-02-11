"""
Training script for VQA model.

This module provides the training loop, validation, and checkpoint management.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from typing import Dict, Optional, Tuple

from dataset import VQADataset, Vocabulary, build_answer_vocab
from model import get_model


class Trainer:
    """
    Trainer class for VQA model training and validation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        device: torch.device,
        checkpoint_dir: str = "../checkpoints",
        print_freq: int = 100,
        grad_clip: float = 5.0,
    ):
        """
        Initialize Trainer.
        
        Args:
            model: VQA model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on (cuda/cpu)
            checkpoint_dir: Directory to save checkpoints
            print_freq: Frequency of printing training stats
            grad_clip: Maximum gradient norm for clipping (0 to disable)
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.print_freq = print_freq
        self.grad_clip = grad_clip
        
        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.best_val_acc = 0.0
    
    def train_epoch(self, epoch: int) -> float:
        """
        Train for one epoch.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        
        # Progress bar
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch} [Train]")
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['images'].to(self.device)
            questions = batch['questions'].to(self.device)
            answers = batch['answers'].to(self.device)
            
            # Forward pass
            # Shape: (batch_size, num_classes)
            logits = self.model(images, questions)
            
            # Compute loss
            loss = self.criterion(logits, answers)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip)
            
            # Update weights
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            # Update progress bar
            if batch_idx % self.print_freq == 0:
                pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
        
        return avg_loss
    
    def validate(self, epoch: int) -> Tuple[float, float]:
        """
        Validate the model.
        
        Args:
            epoch: Current epoch number
        
        Returns:
            Tuple of (average validation loss, accuracy)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch} [Val]")
            
            for batch_idx, batch in enumerate(pbar):
                # Move data to device
                images = batch['images'].to(self.device)
                questions = batch['questions'].to(self.device)
                answers = batch['answers'].to(self.device)
                
                # Forward pass
                logits = self.model(images, questions)
                
                # Compute loss
                loss = self.criterion(logits, answers)
                total_loss += loss.item()
                
                # Compute accuracy
                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == answers).sum().item()
                total += answers.size(0)
                
                # Update progress bar
                avg_loss = total_loss / (batch_idx + 1)
                accuracy = 100.0 * correct / total
                pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{accuracy:.2f}%'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(
        self,
        epoch: int,
        val_acc: float,
        is_best: bool = False,
    ) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            val_acc: Validation accuracy
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
        }
        
        # Save latest checkpoint
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint_latest.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'checkpoint_best.pth')
            torch.save(checkpoint, best_path)
            print(f"Saved best model with validation accuracy: {val_acc:.2f}%")
    
    def load_checkpoint(self, checkpoint_path: str) -> int:
        """
        Load checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Epoch number to resume from
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])
        self.best_val_acc = checkpoint.get('val_acc', 0.0)
        
        epoch = checkpoint['epoch']
        print(f"Loaded checkpoint from epoch {epoch}")
        
        return epoch
    
    def train(
        self,
        num_epochs: int,
        start_epoch: int = 0,
        scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train
            start_epoch: Starting epoch (for resuming training)
            scheduler: Learning rate scheduler (optional)
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        for epoch in range(start_epoch, num_epochs):
            epoch_start_time = time.time()
            
            # Train for one epoch
            train_loss = self.train_epoch(epoch + 1)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss, val_acc = self.validate(epoch + 1)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Update learning rate
            if scheduler:
                scheduler.step()
            
            # Save checkpoint
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            self.save_checkpoint(epoch + 1, val_acc, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{num_epochs} - "
                  f"Time: {epoch_time:.2f}s - "
                  f"Train Loss: {train_loss:.4f} - "
                  f"Val Loss: {val_loss:.4f} - "
                  f"Val Acc: {val_acc:.2f}%\n")


def get_transforms(mode: str = 'train') -> transforms.Compose:
    """
    Get image preprocessing transforms.
    
    Args:
        mode: 'train' or 'val'
    
    Returns:
        Composed transforms
    """
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    else:  # val or test
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])


def main():
    """
    Main training function.
    
    This is a skeleton that needs to be configured with actual data paths
    and hyperparameters.
    """
    # Hyperparameters
    EMBED_SIZE = 512
    HIDDEN_SIZE = 512
    NUM_LAYERS = 1
    DROPOUT = 0.5
    BATCH_SIZE = 64
    NUM_EPOCHS = 30
    LEARNING_RATE = 0.001
    TOP_K_ANSWERS = 1000  # Number of most common answers to consider
    
    # Paths (modify these according to your setup)
    TRAIN_IMAGE_DIR = "../data/train2014"
    VAL_IMAGE_DIR = "../data/val2014"
    TRAIN_QUESTIONS = "../data/v2_OpenEnded_mscoco_train2014_questions.json"
    TRAIN_ANNOTATIONS = "../data/v2_mscoco_train2014_annotations.json"
    VAL_QUESTIONS = "../data/v2_OpenEnded_mscoco_val2014_questions.json"
    VAL_ANNOTATIONS = "../data/v2_mscoco_val2014_annotations.json"
    CHECKPOINT_DIR = "../checkpoints"
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # TODO: Build vocabularies (question vocab and answer vocab)
    # This should be done once and saved/loaded for consistency
    print("Building vocabularies...")
    # question_vocab = Vocabulary(freq_threshold=5)
    # answer_to_idx, idx_to_answer = build_answer_vocab(TRAIN_ANNOTATIONS, top_k=TOP_K_ANSWERS)
    
    # TODO: Create datasets and dataloaders
    print("Creating datasets...")
    # train_dataset = VQADataset(
    #     image_dir=TRAIN_IMAGE_DIR,
    #     questions_file=TRAIN_QUESTIONS,
    #     annotations_file=TRAIN_ANNOTATIONS,
    #     vocab=question_vocab,
    #     transform=get_transforms('train'),
    #     answer_to_idx=answer_to_idx,
    # )
    # 
    # val_dataset = VQADataset(
    #     image_dir=VAL_IMAGE_DIR,
    #     questions_file=VAL_QUESTIONS,
    #     annotations_file=VAL_ANNOTATIONS,
    #     vocab=question_vocab,
    #     transform=get_transforms('val'),
    #     answer_to_idx=answer_to_idx,
    # )
    # 
    # train_loader = DataLoader(
    #     train_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=True,
    #     num_workers=4,
    #     collate_fn=VQADataset.collate_fn,
    # )
    # 
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     collate_fn=VQADataset.collate_fn,
    # )
    
    # TODO: Create model
    print("Creating model...")
    # model = get_model(
    #     vocab_size=len(question_vocab),
    #     num_classes=TOP_K_ANSWERS,
    #     embed_size=EMBED_SIZE,
    #     hidden_size=HIDDEN_SIZE,
    #     num_layers=NUM_LAYERS,
    #     dropout=DROPOUT,
    #     pretrained=True,
    # ).to(device)
    
    # TODO: Define loss and optimizer
    # criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # TODO: Create trainer and start training
    # trainer = Trainer(
    #     model=model,
    #     train_loader=train_loader,
    #     val_loader=val_loader,
    #     criterion=criterion,
    #     optimizer=optimizer,
    #     device=device,
    #     checkpoint_dir=CHECKPOINT_DIR,
    # )
    # 
    # trainer.train(num_epochs=NUM_EPOCHS, scheduler=scheduler)
    
    print("Training skeleton is ready. Uncomment and configure the TODOs to start training.")


if __name__ == '__main__':
    main()
