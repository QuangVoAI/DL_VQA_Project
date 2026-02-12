# DL_VQA_Project

**Midterm Project of XuanQuangVo (523H0173) and XuanThanhHoang (523H0178)**

A comprehensive Visual Question Answering (VQA) system built with PyTorch that combines CNN-based image encoding with LSTM-based question processing to answer natural language questions about images.

## 📋 Overview

This project implements a VQA model on the COCO dataset that:
- Processes visual information using ResNet-50 as an encoder
- Encodes questions using LSTM-based decoder
- Classifies answers from the top-K most common responses
- Handles automatic vocabulary building from question data
- Supports multi-GPU training (CUDA, Apple Silicon MPS, CPU fallback)

**Dataset Stats:**
- Total samples: 214,354
- Training set: 171,483 (80%)
- Validation set: 21,435 (10%)
- Test set: 21,436 (10%)
- Vocabulary size: ~11,000 words
