# MedNCA High-Resolution Training Pipeline

This project contains the implementation and training pipeline for the **MedNCAHighRes** model applied to medical image segmentation, particularly on the ISIC 2018 dataset. The model leverages neural cellular automata (NCA) concepts for evolving fine-grained, pixel-level predictions.

---

## 📁 Directory Structure

mednca/
├── datasets/
│   └── isic_dataset.py           # Custom dataset class for ISIC 2018
├── models/
│   └── mednca_highres.py         # MedNCA high-res model implementation
├── losses.py                     # Combined Dice + Focal loss function
├── train.py                      # Main training script
├── output/
│   └── training_*/               # Output directory for logs, checkpoints, visualizations
├── data/
│   ├── ISIC2018_Task1-2_Training_Input/         # Input skin images
│   └── ISIC2018_Task1_Training_GroundTruth/     # Corresponding masks
└── README.txt                    # This file (rename to README.md)

---

## ⚙️ Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- numpy
- tqdm

Install via:
pip install -r requirements.txt


3. Outputs will be saved under `output/training_<timestamp>/`:
   - Checkpoints (`.pth`)
   - Training loss curves
   - Sample predictions as `.png` files
   - Config used for training (`config.txt`)

---

## 🧠 Model Summary

- Based on **Neural Cellular Automata (NCA)**
- Uses:
  - Patch-based update logic
  - Fire rate stochasticity
  - Hidden states with 16 channels
- High-resolution support: processes full 128×128 image patches
- Built for 1-channel (grayscale) inputs

---

## 📊 Training Features

- Mixed Dice + Focal loss for handling class imbalance
- Adaptive learning rate with `ReduceLROnPlateau`
- Real-time visualizations of predictions
- Robust error handling (e.g., CUDA OOM, NaN loss detection)
- Gradient clipping and GPU memory management

---

## 🧪 Dataset Used

- [ISIC 2018 Skin Lesion Segmentation Dataset](https://challenge2018.isic-archive.com/)
- Images resized to 128×128
- Binary masks used for lesion segmentation

## 👨‍💻 Author

Arnav Kulkarni  
IIT Mandi  
GitHub: [arnavkulkarni2005](https://github.com/arnavkulkarni2005)

---


