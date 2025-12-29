"""
Homoglyph Discovery Script

This script uses the trained Autoencoder to discover homoglyphs (visually similar characters).
It works by:
1. Loading the trained model and image dataset.
2. Computing the latent embedding for every character image.
3. Calculating the pairwise similarity matrix for all embeddings (using Cosine Similarity / Dot Product).
4. Identifying pairs with very high similarity (> 0.95).
5. Filtering out "Tofu" or "Generic" matches (images that match too many distinct characters).
6. Saving the discovered pairs to a CSV file.

The script uses batched PyTorch operations to handle large datasets efficiently.
"""

import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ConvAutoencoder
from train import CharDataset, DATA_DIR, MODEL_SAVE_PATH, DEVICE
import csv
from collections import defaultdict

def discover():
    """
    Main discovery function.
    Loads model, computes embeddings, finds nearest neighbors, filters results, and saves to CSV.
    """
    # Load Model
    model = ConvAutoencoder().to(DEVICE)
    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"Model file {MODEL_SAVE_PATH} not found. Train the model first.")
        return
    model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=DEVICE))
    model.eval()

    # Load Data
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = CharDataset(DATA_DIR, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    print("Computing embeddings...")
    embeddings = []
    filenames = []
    
    # We need to map indices back to filenames/metadata
    # dataset.image_paths matches the order if shuffle=False
    filenames = [os.path.basename(p) for p in dataset.image_paths]

    # Generate embeddings for all images
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            emb = model.encode(data)
            embeddings.append(emb.cpu())
    
    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings = torch.cat(embeddings, dim=0) # (N, latent_dim)
    
    # Normalize embeddings so dot product equals cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)

    # Helper function to extract character from filename "FontName_hex.png"
    def get_char_from_filename(fname):
        try:
            hex_part = fname.rsplit('_', 1)[1].split('.')[0]
            return chr(int(hex_part, 16))
        except:
            return "?"

    chars = [get_char_from_filename(f) for f in filenames]
    
    
    # Optimized search using PyTorch operations with batching to avoid OOM
    print(f"Finding pairs for {len(filenames)} images...")
    threshold = 0.95
    results = []
    
    # Process similarity matrix computation in chunks
    chunk_size = 1000
    N = len(embeddings)
    
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        
        # Current chunk of queries: (chunk_size, Latent)
        queries = embeddings[i_start:i_end]
        
        # Similarity for this chunk: (chunk_size, N)
        # sim[k, j] is similarity between query[k] and embedding[j]
        chunk_sim = torch.mm(queries, embeddings.t())
        
        # Find indices where similarity > threshold
        # items are (k, j) relative to chunk
        matches = torch.nonzero(chunk_sim > threshold, as_tuple=False)
        
        if len(matches) == 0:
            continue

        # Convert to CPU for explicit filtering and logic 
        matches = matches.cpu().numpy()
        
        # 1. Group matches by query index k to analyze per-image behavior
        matches_per_k = defaultdict(list)
        for k, j in matches:
            matches_per_k[k].append(j)
            
        # 2. Iterate and filter for Tofu / Generic glyphs
        for k, j_indices in matches_per_k.items():
            global_i = i_start + k
            
            # Tofu Filter Heuristic: 
            # If one image matches > 5 distinct characters, it's likely a generic "missing glyph" box.
            # Real homoglyphs (e.g., '1', 'l', 'I') usually match only 2-3 other distinct characters.
            
            # Optimization: check length first. If len(j_indices) > 500, it's almost certainly Tofu.
            if len(j_indices) > 500:
                continue
            
            matched_chars = set()
            valid_pairs = []

            for j in j_indices:
                c_j = chars[j]
                matched_chars.add(c_j)
                valid_pairs.append(j)
             
            # If it matches too many distinct characters, skip this query image entirely.
            if len(matched_chars) > 5:
                continue

            # If passed the filter, record the valid pairs
            for j in valid_pairs:
                if j > global_i: # enforce upper triangle to avoid duplicates (A-B vs B-A) and self-matches
                    score = chunk_sim[k, j].item()
                    char_i = chars[global_i]
                    char_j = chars[j]
                
                    # Final check: we are only interested if the CHARACTERS are different.
                    # (Matching 'a' to 'a' in a different font is generic similarity, not a homoglyph)
                    if char_i != char_j:
                         results.append({
                            "char1": char_i,
                            "file1": filenames[global_i],
                            "char2": char_j,
                            "file2": filenames[j],
                            "score": score
                        })

        if i_start % 5000 == 0:
            print(f"Processed up to index {i_start}/{N}, found {len(results)} pairs so far...")


    # Sort results by similarity score (highest first)
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"Found {len(results)} potential homoglyph pairs.")
    
    # Save results to CSV
    with open("homoglyphs_found.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["char1", "file1", "char2", "file2", "score"])
        writer.writeheader()
        writer.writerows(results)
        
    # Print top 10 for quick verification
    for r in results[:10]:
        print(f"{r['char1']} <-> {r['char2']} (Score: {r['score']:.4f}) [{r['file1']} vs {r['file2']}]")

if __name__ == "__main__":
    discover()
