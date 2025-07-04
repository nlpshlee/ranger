import os
import json

from datasets import Dataset, load_dataset
from tqdm import tqdm


def convert_jsonl_to_arrow_stream(jsonl_path: str, save_dir: str, buffer_size: int = 100_000):
    buffer = []
    total = 0
    shard_idx = 0

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                buffer.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}")
                continue

            if len(buffer) >= buffer_size:
                dataset = Dataset.from_list(buffer)
                shard_path = f"{save_dir}_shard{shard_idx}"
                dataset.save_to_disk(shard_path)
                print(f"Saved shard {shard_idx} with {len(buffer)} examples to {shard_path}")
                buffer.clear()
                shard_idx += 1
                total += buffer_size

    # 남은 데이터 저장
    if buffer:
        dataset = Dataset.from_list(buffer)
        shard_path = f"{save_dir}_shard{shard_idx}"
        dataset.save_to_disk(shard_path)
        print(f"Saved final shard {shard_idx} with {len(buffer)} examples to {shard_path}")
        total += len(buffer)

    print(f"Total examples saved: {total}")


def convert_arrow_stream_to_jsonl(file_path: str, save_dir: str):
    dataset = load_dataset(file_path, split="train")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "kilt-corpus.jsonl"), "w", encoding="utf-8") as f:
        for data in tqdm(dataset):
            id = data['id']
            contents, title = data['contents'], data['title']
            if contents.startswith(title + '\n'):
                contents = contents[len(title) + 1:]
            contents = f'{title}\n{contents}'.strip()

            result_dict = {
                "id": id,
                "contents": contents,
            }
            f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    convert_arrow_stream_to_jsonl(
        file_path='corag/kilt-corpus', save_dir='/home/stv10121/saltlux/data'
    )