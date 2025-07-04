from pyserini.search.lucene import LuceneSearcher
import torch
from typing import List, Dict, Union
from src.data_utils import load_corpus


class BM25Searcher:
    def __init__(self, index_dir: str="/home/stv10121/saltlux/data/index/bm25"):
        self.searcher = LuceneSearcher(index_dir)
        self.corpus = load_corpus()

    def _search(self, query: str, top_k: int=5):
        hits = self.searcher.search(query, k=top_k)
        
        doc_ids = [hit.docid for hit in hits]
        return doc_ids