"""
Utility Functions for VQA Project

This module contains helper functions for visualization, evaluation,
and other common tasks.
"""

import os
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


def plot_training_curves(train_losses: List[float],
                         val_losses: List[float],
                         save_path: str = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses (List[float]): List of training losses per epoch
        val_losses (List[float]): List of validation losses per epoch
        save_path (str, optional): Path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Plot saved to {save_path}')
    else:
        plt.show()
    
    plt.close()


def visualize_sample(image: Image.Image,
                    question: str,
                    predicted_answer: str,
                    ground_truth_answer: str = None,
                    save_path: str = None):
    """
    Visualize a VQA sample with image, question, and answers.
    
    Args:
        image (PIL.Image): Input image
        question (str): Question text
        predicted_answer (str): Model's predicted answer
        ground_truth_answer (str, optional): Ground truth answer
        save_path (str, optional): Path to save visualization
    """
    plt.figure(figsize=(10, 8))
    
    # Display image
    plt.imshow(image)
    plt.axis('off')
    
    # Add text
    title = f'Question: {question}\n'
    title += f'Predicted: {predicted_answer}'
    if ground_truth_answer:
        title += f'\nGround Truth: {ground_truth_answer}'
    
    plt.title(title, fontsize=12, pad=20, wrap=True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f'Visualization saved to {save_path}')
    else:
        plt.show()
    
    plt.close()


def calculate_accuracy(predictions: List[str],
                      ground_truths: List[str]) -> float:
    """
    Calculate exact match accuracy.
    
    Args:
        predictions (List[str]): List of predicted answers
        ground_truths (List[str]): List of ground truth answers
        
    Returns:
        float: Accuracy percentage
    """
    if len(predictions) != len(ground_truths):
        raise ValueError('Predictions and ground truths must have same length')
    
    correct = sum(1 for pred, gt in zip(predictions, ground_truths)
                  if pred.lower().strip() == gt.lower().strip())
    
    accuracy = (correct / len(predictions)) * 100
    return accuracy


def save_predictions(predictions: List[Dict],
                    output_file: str):
    """
    Save predictions to JSON file.
    
    Args:
        predictions (List[Dict]): List of prediction dictionaries with keys:
            'question_id', 'question', 'predicted_answer', 'ground_truth'
        output_file (str): Path to output JSON file
    """
    with open(output_file, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f'Predictions saved to {output_file}')


def load_predictions(input_file: str) -> List[Dict]:
    """
    Load predictions from JSON file.
    
    Args:
        input_file (str): Path to input JSON file
        
    Returns:
        List[Dict]: List of prediction dictionaries
    """
    with open(input_file, 'r') as f:
        predictions = json.load(f)
    
    return predictions


def denormalize_image(image_tensor: torch.Tensor,
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize image tensor for visualization.
    
    Args:
        image_tensor (torch.Tensor): Normalized image tensor of shape (3, H, W)
        mean (List[float]): Mean used for normalization
        std (List[float]): Standard deviation used for normalization
        
    Returns:
        np.ndarray: Denormalized image array of shape (H, W, 3)
    """
    # Clone to avoid modifying original
    image = image_tensor.clone()
    
    # Denormalize
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    
    # Clamp values to [0, 1]
    image = torch.clamp(image, 0, 1)
    
    # Convert to numpy and transpose to (H, W, C)
    image_np = image.cpu().numpy().transpose(1, 2, 0)
    
    return image_np


def count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """
    Count total and trainable parameters in a model.
    
    Args:
        model (torch.nn.Module): PyTorch model
        
    Returns:
        Tuple[int, int]: (total_params, trainable_params)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def setup_directories(dirs: List[str]):
    """
    Create directories if they don't exist.
    
    Args:
        dirs (List[str]): List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
        print(f'Directory created/verified: {dir_path}')


def get_device(device_id: int = None) -> torch.device:
    """
    Get PyTorch device (GPU if available, else CPU).
    
    Args:
        device_id (int, optional): Specific GPU device ID
        
    Returns:
        torch.device: PyTorch device
    """
    if torch.cuda.is_available():
        if device_id is not None:
            device = torch.device(f'cuda:{device_id}')
        else:
            device = torch.device('cuda')
        print(f'Using device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        device = torch.device('cpu')
        print('Using device: CPU')
    
    return device


class AverageMeter:
    """
    Computes and stores the average and current value.
    Useful for tracking metrics during training.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update statistics.
        
        Args:
            val (float): New value
            n (int): Number of samples
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_model_summary(model: torch.nn.Module):
    """
    Print a summary of the model architecture.
    
    Args:
        model (torch.nn.Module): PyTorch model
    """
    print('\n' + '='*70)
    print('MODEL SUMMARY')
    print('='*70)
    print(model)
    print('='*70)
    
    total_params, trainable_params = count_parameters(model)
    print(f'Total parameters: {total_params:,}')
    print(f'Trainable parameters: {trainable_params:,}')
    print(f'Non-trainable parameters: {total_params - trainable_params:,}')
    print('='*70 + '\n')


def save_config(config: Dict, output_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config (Dict): Configuration dictionary
        output_path (str): Path to save configuration
    """
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f'Configuration saved to {output_path}')


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f'Configuration loaded from {config_path}')
    return config
