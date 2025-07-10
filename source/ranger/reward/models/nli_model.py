import torch

from typing import List
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIModel:
    def __init__(
            self, model_name_or_path: str='cross-encoder/nli-deberta-v3-small',
    ):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)


    def _predict(
        self, premise: str, hypothesis: str
    ):
        """
        score: tensor([ 7.2162, -4.1645, -2.4216])
        label: ['contradiction'] # label: contradiction, entailment, neutral
        """
        features = self.tokenizer(
            [premise], [hypothesis],
            padding=True, truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.model(**features).logits
            prob = torch.softmax(scores, dim=1)[0][1].item()  # entailment probability
        return prob
    
    def batch_predict(
        self,
        premises: List[str],
        hypotheses: List[str]
    ):
        features = self.tokenizer(
            premises, hypotheses,
            padding=True, truncation=True, return_tensors="pt"
        )
        
        with torch.no_grad():
            scores = self.model(**features).logits
            probs = torch.softmax(scores, dim=1)[:, 1].tolist()  # entailment probability
        return probs