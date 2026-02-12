import argparse
import os

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict


def make_wiki_documents(elasticsearch_index: str, metadata: Dict = None):
    metadata = metadata or {"idx": 0}
    assert "idx" in metadata

    dataset = load_dataset("corag/kilt-corpus", split="train")
    for data in tqdm(dataset):
        title, contents = data['title'], data['contents']

        es_paragraph = {
            "id": metadata["idx"],
            "title": title,
            "paragraph_text": contents,
        }
        
        document = {
            "_op_type": "create",
            "_index": elasticsearch_index,
            "_id": metadata["idx"],
            "_source": es_paragraph,
        }
        yield (document)
        metadata["idx"] += 1


def main():
    parser = argparse.ArgumentParser(description="Index Paragraphs in Elasticsearch")
    parser.add_argument(
        "--force",
        help="force delete before creating new index.",
        action="store_true",
        default=False
    )
    args = parser.parse_args()

    # connect elasticsearch
    elastic_host = "localhost"
    elastic_port = 9200
    elasticsearch_index = "wiki"
    
    es = Elasticsearch(
        [
            {
                "host": elastic_host,
                "port": elastic_port,
                "scheme": "http"
            }
        ],
        max_retries=20,
        request_timeout=2000,
        retry_on_timeout=True,
    )

    paragraphs_index_settings = {
        "mappings": {
            "properties": {
                "title": {
                    "type": "text",
                    "analyzer": "english"
                },
                "paragraph_text": {
                    "type": "text",
                    "analyzer": "english"
                }
            }
        }
    }

    index_exists = es.indices.exists(index=elasticsearch_index)
    print("Index already exists" if index_exists else "Index does not exist")

    # delete index if exists
    if index_exists:
        if not args.force:
            feedback = input(f"Index {elasticsearch_index} already exists. " f"Are you sure you want to delete it?")
            if not (feedback.startswith("y") or feedback == ""):
                exit("Terminated by user.")
        es.indices.delete(index=elasticsearch_index)

    # create index
    print("Creating Index ...")
    es.indices.create(index=elasticsearch_index, ignore=400, body=paragraphs_index_settings)

    # bulk-insert documents into index
    print("Inserting Paragraphs ...")
    result = bulk(
        es,
        make_wiki_documents(elasticsearch_index=elasticsearch_index),
        raise_on_error=True, # set to true o/w it'll fail silently and only show less docs.
        raise_on_exception=True, # set to true o/w it'll fail silently and only show less docs.
        max_retries=2, # it's exp backoff starting 2, more than 2 retries will be too much.
        request_timeout=500,
    )

    es.indices.refresh(elasticsearch_index)
    document_count = result[0]
    print(f"Index {elasticsearch_index} is ready. Added {document_count} documents.")


if __name__ == "__main__":
    main()