import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from model import ConvAutoencoder
from train import CharDataset, DATA_DIR, MODEL_SAVE_PATH, DEVICE
import csv

def discover():
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

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(DEVICE)
            emb = model.encode(data)
            embeddings.append(emb.cpu())
    
    if not embeddings:
        print("No embeddings generated.")
        return

    embeddings = torch.cat(embeddings, dim=0) # (N, latent_dim)
    
    # Normalize for Cosine Similarity (optional but good for embeddings)
    embeddings = F.normalize(embeddings, p=2, dim=1)

    print(f"Computing similarity matrix for {len(filenames)} images...")
    # Similarity matrix: (N, N)
    # This might be large if N is huge. 
    # If N=1000, 1000x1000 is small. If N=100k, we might need a better approach (FAISS).
    # Assuming < 10k images for now.
    
    sim_matrix = torch.mm(embeddings, embeddings.t())
    
    # Find high similarities off-diagonal
    # We want to find pairs (i, j) where i != j and char(i) != char(j)
    
    # Helper to parse char from filename: "FontName_hex.png"
    def get_char_from_filename(fname):
        try:
            hex_part = fname.rsplit('_', 1)[1].split('.')[0]
            return chr(int(hex_part, 16))
        except:
            return "?"

    chars = [get_char_from_filename(f) for f in filenames]
    
    results = []
    
    # Optimized search using PyTorch operations
    print("Finding pairs...")
    threshold = 0.95
    
    # Mask diagonal and lower triangle to avoid duplicates and self-matches
    mask = torch.triu(torch.ones_like(sim_matrix), diagonal=1).bool()
    
    # Filter by threshold
    # We want sim_matrix > threshold AND mask
    # To save memory, we can iterate in chunks or just try to find indices directly.
    # Since we are on GPU/result is on GPU, let's keep it there.
    
    # Find indices where similarity > threshold
    # (This returns indices on the device)
    # We use a slightly higher threshold to reduce output if too many, but let's stick to 0.95
    pairs = torch.nonzero((sim_matrix > threshold) & mask, as_tuple=False)
    
    print(f"Processing {len(pairs)} pairs...")
    
    results = []
    
    # Move chars/filenames to a lookup that is fast
    # (Already lists)
    
    # Convert pairs to CPU list for iteration (much smaller than N*N)
    pairs_cpu = pairs.cpu().numpy()
    
    for i, j in pairs_cpu:
        score = sim_matrix[i, j].item()
        char_i = chars[i]
        char_j = chars[j]
        
        if score > threshold: # Double check (redundant but safe)
            if char_i != char_j:
                 results.append({
                    "char1": char_i,
                    "file1": filenames[i],
                    "char2": char_j,
                    "file2": filenames[j],
                    "score": score
                })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"Found {len(results)} potential homoglyph pairs.")
    
    # Save results
    with open("homoglyphs_found.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["char1", "file1", "char2", "file2", "score"])
        writer.writeheader()
        writer.writerows(results)
        
    # Print top 10
    for r in results[:10]:
        print(f"{r['char1']} <-> {r['char2']} (Score: {r['score']:.4f}) [{r['file1']} vs {r['file2']}]")

if __name__ == "__main__":
    discover()
