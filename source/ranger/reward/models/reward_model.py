import numpy as np
import statsmodels.api as sm

from typing import List
from sklearn.metrics import roc_auc_score, roc_curve, f1_score
from sklearn.model_selection import train_test_split
from ranger.corag.agent.agent_utils import RagPath
from ranger.reward.models.feature_extractor import FeatureExtractor
from ranger.reward.data_utils import load_json_file


class RewardModel(FeatureExtractor):
    def __init__(self, params_path: str):
        # initialize FeatureExtractor
        super().__init__()
        
        # load parameters
        self._params_path = params_path
        self.coef, self.intercept, self.best_threshold = self.load_params()


    def load_params(self):
        data = load_json_file(self._params_path)
        coef, intercept, best_threshold = data['coef'], data['intercept'], data['best_threshold']
        return coef, intercept, best_threshold
    

    def trained_LR_model(self, X, y):
        # Split the data into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Add a constant term to the features for the intercept for training and testing set
        X_train = sm.add_constant(X_train)
        X_test = sm.add_constant(X_test)

        # Fit the logistic regression model using statsmodels
        model = sm.Logit(y_train, X_train)
        result = model.fit()
        print(result.summary())

        # update intercept and coef
        params = result.params
        intercept, coef = params[0], params[1:]
        self.intercept, self.coef = intercept, coef
        
        # Make predictions on the test data (predicting probabilities)
        y_pred_proba = result.predict(X_test)

        # Calculate the AUROC
        auroc = roc_auc_score(y_test, y_pred_proba)
        print(f"The AUROC score is: {auroc}")
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        f1_scores = [f1_score(y_test, y_pred_proba > thresh) for thresh in thresholds]
        
        # update best threshold
        best_threshold = thresholds[np.argmax(f1_scores)]
        self.best_threshold = best_threshold


    def customized_LR(self, features: List[float]) -> float:
        # evaluate the linear combination of features and coefficients based on the trained parameters
        lincomb = sum([self.coef[i]*features[i] for i in range(len(self.coef))]) + self.intercept
        return 1 / (1 + np.exp(-lincomb))
    

    def calculate_reward(self, ragpath: RagPath) -> float:
        feature_vector = self.extract_feature_vector(ragpath)
        LS, QR, SD, EA = feature_vector['LS'], feature_vector['QR'], feature_vector['SD'], feature_vector['EA']
        reward = self.customized_LR([LS, QR, SD, EA])
        return reward
    
    
    def calculate_reward_batch(self, ragpath_list: List[RagPath]) -> List[float]:
        rewards = [self.calculate_reward(ragpath) for ragpath in ragpath_list]
        return rewards

