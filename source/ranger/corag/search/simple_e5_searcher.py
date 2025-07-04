import torch
import torch.nn.functional as F

from typing import List
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

class SimpleE5Searcher:
    def __init__(self, model_name_or_path: str):
        self.model_name_or_path = model_name_or_path
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.model = AutoModel.from_pretrained(model_name_or_path)

    def search(self, query: str, passages: List[str], top_k=5):
        query = f"query: {query}"
        passages_temp = [f"passage: {passage}" for passage in passages]

        encoded = self.tokenizer(
            [query]+passages_temp,
            max_length=512,
            padding=True,
            truncation=True,
            return_tensors='pt'
        )

        with torch.no_grad():
            outputs = self.model(**encoded)

        embeddings = average_pool(outputs.last_hidden_state, encoded['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        """
        embeddings
        tensor(
            [[-0.0046, -0.0411,  0.0387,  ..., -0.0314,  0.0138,  0.0440], -> query 
            [-0.0016, -0.0689,  0.0052,  ..., -0.0232,  0.0295,  0.0176], -> doc(1)
            [-0.0060, -0.0379,  0.0233,  ..., -0.0312,  0.0246,  0.0458], -> doc(2)
            ...,
            [ 0.0150, -0.0648, -0.0172,  ...,  0.0036,  0.0096,  0.0386],
            [ 0.0032, -0.0766,  0.0228,  ..., -0.0403,  0.0448,  0.0304],
            [-0.0057, -0.0236,  0.0212,  ..., -0.0376,  0.0314,  0.0433]] -> doc(n)
        ) 
        """

        query_embedding = embeddings[0] # (1, hidden_dim)
        passage_embeddings = embeddings[1:] # (# passages, hidden_dim)
        scores = (query_embedding @ passage_embeddings.T) * 100
        """
        scores
        tensor(
            [79.1972, 79.9407, 77.5866, 77.9257, 79.0741, 78.6277, 78.1407, 78.5858,
            77.9638, 78.3098, 77.6256, 79.0896, 77.5707, 78.3446, 76.7430, 77.0530,
            76.6013, 77.4029, 77.8991, 77.4894, 73.5159, 76.8501, 76.6204, 76.8302,
            77.1980, 79.5570, 76.8531, 77.9147, 76.1367, 73.5714, 77.2931, 78.1644,
            76.3222, 76.0493, 76.2629, 75.9206, 77.5705, 76.9622, 75.9542, 76.7015,
            76.6579, 75.7986, 71.3847, 76.6494, 77.0496, 76.3951, 76.6385, 76.2157,
            75.9787, 76.6417, 76.6650, 76.6264, 75.5256, 75.4990, 76.2988, 76.9251,
            76.0149, 75.7544, 75.9274, 75.7916, 77.5429, 75.5300, 77.1941, 76.3726,
            72.6878, 76.0374, 77.2775, 72.1375, 72.9305, 76.9701, 76.3186, 77.7846,
            69.2231, 75.4232, 75.1790, 73.7156, 73.6608, 74.9458, 77.1959, 73.0427,
            76.2304, 76.0567, 76.6128, 75.1835, 76.0768, 77.3939, 76.8786, 76.0988,
            76.1961, 76.6318, 76.5303, 75.4324, 75.3148, 74.6132, 76.2724, 75.5360,
            77.6673, 75.6384, 76.5006, 74.9447]
        )
        """

        top_k_scores, top_k_indices = torch.topk(scores, k=top_k)
        return top_k_indices.tolist()