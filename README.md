# Homoglyph Discovery AI

This project uses Deep Learning (a Convolutional Autoencoder) to discover homoglyphs (characters that look visually similar but have different Unicode values). This is useful for cybersecurity research, specifically for identifying potential IDN homograph attacks, phishing vectors, and typosquatting opportunities.

## How It Works

1.  Rendering: The system renders ASCII characters (and others) using all available system fonts into 64x64 grayscale images.
2.  Learning: A Convolutional Autoencoder learns to compress these images into a low-dimensional "latent space" (embedding).
3.  Discovery: We compute the similarity between all character embeddings. Pairs with high cosine similarity (>0.95) are flagged as homoglyphs.

## Installation

Prerequisites: Python 3.x, `pip`, and standard system fonts.

1.  Create a virtual environment (recommended):
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

2.  Install dependencies:
    ```bash
    pip install numpy Pillow torch torchvision
    ```

## Usage

### 1. Generate Dataset
Render character images from your installed system fonts.
```bash
python3 generate_data.py
```
*Output: `dataset/images/*.png` and `dataset/metadata.csv`*

### 2. Train Model
Train the Autoencoder on the generated images.
```bash
python3 train.py
```
*Output: `homoglyph_model.pth`*

### 3. Discover Homoglyphs
Run the analysis script to find similar characters.
```bash
python3 discover.py
```
*Output: `homoglyphs_found.csv`*

## Files

*   `generate_data.py`: Script to render characters using Pillow.
*   `model.py`: PyTorch definition of the Convolutional Autoencoder.
*   `train.py`: Training loop for the model.
*   `discover.py`: Analysis script. Computes embeddings, finds nearest neighbors, filters generic matches, and outputs results.
*   `homoglyphs_found.csv`: The final list of discovered homoglyph pairs.

## Results

 The output CSV contains columns: char1, file1, char2, file2, score.
 
 Example:
 ```
 I,arialbd.ttf_49.png,l,arialbd.ttf_6c.png,0.9979
 ```
 This indicates that I (uppercase i) and l (lowercase L) in Arial Bold look nearly identical.


## Future Work

Future work in this project would likely include using a larger dataset (more fonts) to gather more data on homoglyphs.
