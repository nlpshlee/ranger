import os
import torch

from tqdm import tqdm
from datasets import load_dataset


# setting
names = [
    "hotpotqa",
    "2wikimultihopqa",
    "musique"
]

input_dir = "/home/stv10121/saltlux/data/e5-large-index/org"
output_dir = "/home/stv10121/saltlux/data/e5-large-index/filtered"

# load used context doc ids
filtered_context_doc_ids = set()
for name in tqdm(names, desc="filter context_doc_ids ..."):
    dataset = load_dataset("corag/multihopqa", name, split="train")
    
    for data in dataset:
        filtered_context_doc_ids.update(map(int, data['context_doc_ids']))

print(f"length of unique context_doc_ids: {len(filtered_context_doc_ids)}")

"""
corag/multihopqa # of unique docs: 4,894,035
"""

# shard 설정
num_output_shards = 8
cut_points = [i*len(filtered_context_doc_ids) // num_output_shards for i in range(1, num_output_shards+1)]

# select embeddings
shard_paths = sorted(
    [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pt") and "shard" in f],
    key=lambda x: int(x.split('-shard-')[1].split('.')[0])
)
print(shard_paths)

shard_buffer = []
shard_idx, global_idx = 0, 0

num_selected = 0
for shard_path in tqdm(shard_paths, desc="filter embeddings..."):
    tensor = torch.load(shard_path, map_location='cpu') # (shard_size, embedding_dim)
    num = tensor.size(0)

    for i in range(num):
        current_idx = global_idx + i
        if current_idx in filtered_context_doc_ids:
            shard_buffer.append(tensor[i].unsqueeze(0))
            num_selected += 1

        if shard_idx < len(cut_points) and num_selected >= cut_points[shard_idx]:
            save_path = os.path.join(output_dir, f"used-shard-{shard_idx}.pt")
            torch.save(torch.cat(shard_buffer, dim=0), save_path)
            print(f"Saved shard-{shard_idx} at index {current_idx}: {len(shard_buffer)} embeddings")
        
            # clear!
            shard_buffer = []
            shard_idx += 1

    global_idx += num

if shard_buffer and shard_idx < num_output_shards:
    save_path = os.path.join(output_dir, f"used-shard-{shard_idx}.pt")
    to_save = torch.save(torch.cat(shard_buffer, dim=0), save_path)

    if to_save.size(0) == 0:
        print(f"[warning] shard-{shard_idx} has 0 embeddings! skip save")
    else:
        torch.save(to_save, save_path)
        print(f"Saved final shard-{shard_idx}: {len(shard_buffer)} embeddings")