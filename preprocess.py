"""
VQA Dataset Preprocessing Script

This script reads VQA v2.0 JSON files and creates a subset with the first N images.
Useful for creating smaller datasets for faster experimentation.

Usage:
    python preprocess.py --num_images 5000
"""

import json
import os
import argparse
from collections import defaultdict
from typing import Dict, List, Set


def load_json(file_path: str) -> dict:
    """Load JSON file."""
    print(f"Loading {file_path}...")
    with open(file_path, 'r') as f:
        return json.load(f)


def save_json(data: dict, file_path: str) -> None:
    """Save data to JSON file."""
    print(f"Saving to {file_path}...")
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved successfully!")


def create_subset(
    questions_file: str,
    annotations_file: str,
    output_questions_file: str,
    output_annotations_file: str,
    num_images: int = 5000
) -> None:
    """
    Create a subset of VQA dataset with the first N images.
    
    Args:
        questions_file: Path to original questions JSON file
        annotations_file: Path to original annotations JSON file
        output_questions_file: Path to save subset questions JSON
        output_annotations_file: Path to save subset annotations JSON
        num_images: Number of images to include in subset
    """
    # Load original data
    questions_data = load_json(questions_file)
    annotations_data = load_json(annotations_file)
    
    # Extract questions and annotations lists
    questions = questions_data.get('questions', [])
    annotations = annotations_data.get('annotations', [])
    
    print(f"\nOriginal dataset:")
    print(f"  Total questions: {len(questions)}")
    print(f"  Total annotations: {len(annotations)}")
    
    # Get unique image IDs from questions (maintaining order)
    image_ids_seen = []
    image_id_set: Set[int] = set()
    
    for q in questions:
        img_id = q['image_id']
        if img_id not in image_id_set:
            image_ids_seen.append(img_id)
            image_id_set.add(img_id)
            if len(image_ids_seen) >= num_images:
                break
    
    # Select first N unique images
    selected_image_ids = set(image_ids_seen[:num_images])
    
    print(f"\nSelected {len(selected_image_ids)} unique images")
    
    # Filter questions for selected images
    subset_questions = [
        q for q in questions 
        if q['image_id'] in selected_image_ids
    ]
    
    # Create question_id set for filtering annotations
    subset_question_ids = {q['question_id'] for q in subset_questions}
    
    # Filter annotations for selected questions
    subset_annotations = [
        ann for ann in annotations 
        if ann['question_id'] in subset_question_ids
    ]
    
    print(f"\nSubset dataset:")
    print(f"  Images: {len(selected_image_ids)}")
    print(f"  Questions: {len(subset_questions)}")
    print(f"  Annotations: {len(subset_annotations)}")
    
    # Create output data structure (preserve original format)
    output_questions_data = {
        'info': questions_data.get('info', {}),
        'task_type': questions_data.get('task_type', ''),
        'data_type': questions_data.get('data_type', ''),
        'data_subtype': questions_data.get('data_subtype', ''),
        'questions': subset_questions
    }
    
    output_annotations_data = {
        'info': annotations_data.get('info', {}),
        'license': annotations_data.get('license', {}),
        'data_type': annotations_data.get('data_type', ''),
        'data_subtype': annotations_data.get('data_subtype', ''),
        'annotations': subset_annotations
    }
    
    # Save subset files
    save_json(output_questions_data, output_questions_file)
    save_json(output_annotations_data, output_annotations_file)
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Subset creation completed!")
    print(f"{'='*60}")
    print(f"Questions saved to: {output_questions_file}")
    print(f"Annotations saved to: {output_annotations_file}")
    print(f"\nYou can now use these files with VQADataset class.")


def main():
    parser = argparse.ArgumentParser(
        description='Create VQA dataset subset with first N images'
    )
    
    # Input files
    parser.add_argument(
        '--questions',
        type=str,
        default='data/v2_OpenEnded_mscoco_train2014_questions.json',
        help='Path to original questions JSON file'
    )
    parser.add_argument(
        '--annotations',
        type=str,
        default='data/v2_mscoco_train2014_annotations.json',
        help='Path to original annotations JSON file'
    )
    
    # Output files
    parser.add_argument(
        '--output_questions',
        type=str,
        default='data/subset_questions.json',
        help='Path to save subset questions JSON'
    )
    parser.add_argument(
        '--output_annotations',
        type=str,
        default='data/subset_annotations.json',
        help='Path to save subset annotations JSON'
    )
    
    # Subset size
    parser.add_argument(
        '--num_images',
        type=int,
        default=5000,
        help='Number of images to include in subset (default: 5000)'
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    if not os.path.exists(args.questions):
        print(f"Error: Questions file not found: {args.questions}")
        print("Please download VQA v2.0 dataset first.")
        return
    
    if not os.path.exists(args.annotations):
        print(f"Error: Annotations file not found: {args.annotations}")
        print("Please download VQA v2.0 dataset first.")
        return
    
    # Create output directories if they don't exist
    for output_file in [args.output_questions, args.output_annotations]:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
    
    # Create subset
    create_subset(
        questions_file=args.questions,
        annotations_file=args.annotations,
        output_questions_file=args.output_questions,
        output_annotations_file=args.output_annotations,
        num_images=args.num_images
    )


if __name__ == '__main__':
    main()
