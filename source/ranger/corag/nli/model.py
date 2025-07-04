from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class NLIModel:
    def __init__(
            self, model_name_or_path: str='cross-encoder/nli-deberta-v3-base',
        ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def _predict(
            self, premise: List[str], hypothesis: List[str]
    ):
        """
        score: tensor([ 7.2162, -4.1645, -2.4216])
        label: ['contradiction'] # label: contradiction, entailment, neutral
        """
        features = self.tokenizer(
            premise, hypothesis,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.model(**features).logits
            label_mapping = ['contradiction', 'entailment', 'neutral']
            labels = [label_mapping[score_max] for score_max in scores.argmax(dim=1)]
        return scores.max().item(), labels