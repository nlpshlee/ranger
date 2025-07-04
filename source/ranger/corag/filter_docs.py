import os
import json

from tqdm import tqdm
from datasets import load_dataset


# setting
names = [
    "hotpotqa",
    "2wikimultihopqa",
    "musique"
]

output_dir = "/home/stv10121/saltlux/data"
context_docs_used_path = os.path.join(output_dir, "kilt-corpus-used.jsonl")
context_docs_unused_path = os.path.join(output_dir, "kilt-corpus-unused.jsonl")

# load used context doc ids
used_context_doc_ids = set()
for name in tqdm(names, desc="filter context_doc_ids ..."):
    dataset = load_dataset("corag/multihopqa", name, split="train")
    
    for data in dataset:
        used_context_doc_ids.update(map(int, data['context_doc_ids']))
print(f"length of unique context_doc_ids: {len(used_context_doc_ids)}")

"""
corag/multihopqa # of unique docs: 4,894,035
"""

corpus = load_dataset('corag/kilt-corpus', split='train')
with open(context_docs_used_path, 'w', encoding='utf-8') as out_f1, \
    open(context_docs_unused_path, 'w', encoding='utf-8') as out_f2:

    for item in tqdm(corpus, desc='splitting corpus ...'):
        result_dict = { 
            "id_org": item['id'],
            "contents": item['contents'],
            "title": item['title'],
            "wikipedia_id": item['wikipedia_id']
        }

        # save
        json_line = json.dumps(result_dict, ensure_ascii=False) + '\n'
        if int(item['id']) in used_context_doc_ids:
            out_f1.write(json_line)
        else:
            out_f2.write(json_line)