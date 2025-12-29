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
    

    
    # Optimized search using PyTorch operations with batching to avoid OOM
    print(f"Finding pairs for {len(filenames)} images...")
    threshold = 0.95
    results = []
    
    # Process in chunks
    chunk_size = 1000
    N = len(embeddings)
    
    for i_start in range(0, N, chunk_size):
        i_end = min(i_start + chunk_size, N)
        
        # Current chunk of queries: (chunk, Latent)
        queries = embeddings[i_start:i_end]
        
        # Similarity for this chunk: (chunk, N)
        # sim[k, j] is similarity between query[k] and embedding[j]
        # where query[k] corresponds to global index i_start + k
        chunk_sim = torch.mm(queries, embeddings.t())
        
        # We only care about j > i (upper triangle)
        # Create a mask for valid j indices
        
        # Global indices for rows in this chunk: [i_start, ..., i_end-1]
        # We want j > i.
        
        # Let's find indices where sim > threshold
        # items are (k, j) relative to chunk
        matches = torch.nonzero(chunk_sim > threshold, as_tuple=False)
        

        
        # Convert to CPU for explicit logic (vectorized filtering above was insufficient)
        matches = matches.cpu().numpy()
        
        # 1. Group matches by query index k
        from collections import defaultdict
        matches_per_k = defaultdict(list)
        for k, j in matches:
            matches_per_k[k].append(j)
            
        # 2. Iterate and filter
        for k, j_indices in matches_per_k.items():
            global_i = i_start + k
            
            # Heuristic: If one image matches > 5 distinct characters, it's garbage/tofu.
            # (e.g. 'l' matches '1', 'I', '|' -> 3 chars. Safe.
            #  Tofu matches 'a', 'b', ... 'z' -> 26 chars. Filtered.)
            
            matched_chars = set()
            valid_pairs = []
            
            # Optimization: check length first. If len(j_indices) > 500, it's almost certainly Tofu.
            # (Unless we have > 500 fonts installed with identical 'l' and '1', which is possible but unlikely to be *exact* matches)
            if len(j_indices) > 500:
                continue
            
            for j in j_indices:
                c_j = chars[j]
                matched_chars.add(c_j)
                valid_pairs.append(j)
                
            if len(matched_chars) > 5:
                continue

            # If passed, add to results
            for j in valid_pairs:
                if j > global_i: # enforce upper triangle
                    score = chunk_sim[k, j].item()
                    char_i = chars[global_i]
                    char_j = chars[j]
                
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
