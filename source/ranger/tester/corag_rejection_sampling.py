import json, string, re
from tqdm import tqdm
from datasets import load_dataset

from ranger.corag.agent.new_corag_agent import CoRagAgent
from ranger.corag.agent.agent_utils import RagPath
from ranger.corag.search.e5_searcher import E5Searcher
from ranger.corag.data_utils import format_input_context
from ranger.corag.logger_config import logger
from ranger.corag.search.bm25_searcher import BM25Searcher


MAX_CHAIN_LENGTH = 6


def _normalize_answer(text, punc_chars=string.punctuation, punc_repl=""):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(s):
        return re.sub(r"\b(a|an|the)\b", " ", s)

    def replace_punctuation(s):
        to_replace = set(punc_chars)
        return "".join(punc_repl if ch in to_replace else ch for ch in s)

    def white_space_fix(s):
        return " ".join(s.split())

    text = text.lower()
    text = replace_punctuation(text)
    text = remove_articles(text)
    text = white_space_fix(text)

    return text


def main():
    # setup
    config = {
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "retriever_name": "intfloat/e5-large-v2",
        "task_desc": "answer multi-hop questions",
    }
    
    # LLM
    corag_agent = CoRagAgent(
        model_name=config["model_name"],
        task_desc=config["task_desc"]
    )

    # Retriever
    retriever = BM25Searcher()

    # load dataset
    dataset = load_dataset("corag/multihopqa", "hotpotqa", split="train").shuffle(seed=42).select(range(100))
    
    # rejection sampling
    with open("/home/stv10121/saltlux/data/output/hotpotqa_samples_100.jsonl", 'w', encoding='utf-8') as out_f:
        for data in tqdm(dataset):
            query_id = data['query_id']
            query = data['query']
            answers = data['answers']

            result_dict = {
                "query_id": query_id,
                "query": query,
                "chains": [],
                "answers": answers,
            }

            answers = ''.join(answers).strip()
            for i in range(16):
                chain = {
                    "query": query,
                    "past_subqueries": [],
                    "past_subanswers": [],
                    "past_doc_ids": [],
                }
                
                logger.info(f"query: {query}")
                for j in range(MAX_CHAIN_LENGTH):
                    
                    """
                    field 정보
                    {
                        "query_id": str
                        "query": str,
                        "subqueries": List[str],
                        "subanswers": List[str],
                        "answers": str,
                        "context_doc_ids": List[str]
                    }
                    """
                    
                    # generate subquery
                    subquery = corag_agent.generate_subquery(query, **chain)
                    chain['past_subqueries'].append(subquery)
                    logger.info(f"subquery: {subquery}")
                    
                    # retrieve docs
                    doc_ids = retriever._search(subquery, 5)
                    chain['past_doc_ids'].append(doc_ids)
                    docs = [format_input_context(retriever.corpus[int(doc_id)]) for doc_id in doc_ids]

                    # generate subanswer
                    subanswer = corag_agent.generate_subanswer(subquery, docs)
                    chain['past_subanswers'].append(subanswer)
                    logger.info(f"subanswer: {subanswer}")
                    
                    if _normalize_answer(subanswer) == _normalize_answer(answers):
                        break
                
                # calculate log likelihood
                log_likelihood = corag_agent.compute_log_likelihood(
                    chain=chain,
                    answers=[answers]
                )
                logger.info(f"log likelihood: {log_likelihood}")

                # retrieve docs for original query
                doc_ids = retriever._search(query, 5)
                docs = [format_input_context(retriever.corpus[int(doc_id)]) for doc_id in doc_ids]

                # generate final answer
                prediction = corag_agent.generate_final_answer( 
                    corag_sample=chain,
                    documents=docs
                )
                logger.info(f"prediction: {prediction}")
                logger.info(f"answers: {answers}")

                # add chain info
                result_dict["chains"].append({
                    "subqueries": chain['past_subqueries'],
                    "subanswers": chain['past_subanswers'],
                    "doc_ids": chain['past_doc_ids'],
                    "log_likelihood": log_likelihood,
                    "prediction": prediction,
                    "label": 1 if _normalize_answer(prediction) == _normalize_answer(answers) else 0
                })

            # write result
            out_f.write(json.dumps(result_dict, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()