"""
프롬프트 전처리를 위한 모듈
"""

import torch
from typing import Dict, List, Any, Union
from transformers import PreTrainedTokenizerBase

from trl.data_utils import maybe_apply_chat_template


class PromptProcessor:
    """
    GRPO 훈련을 위한 프롬프트 전처리 클래스
    """
    
    def __init__(self, processing_class: PreTrainedTokenizerBase, max_prompt_length: int = None):
        """
        Args:
            processing_class: 토크나이저 클래스(객체)
            max_prompt_length: 최대 프롬프트 길이 (None이면 제한 없음)
        """
        self.processing_class = processing_class
        self.max_prompt_length = max_prompt_length
    
    def process_prompts(self, inputs: List[Dict[str, Any]], device: torch.device) -> Dict[str, torch.Tensor]:
        """
        프롬프트를 전처리하여 모델 입력에 필요한 텐서들을 반환합니다.
        
        Args:
            inputs: 입력 데이터 리스트
            device: 텐서를 배치할 디바이스
            
        Returns:
            전처리된 프롬프트 텐서들을 포함한 딕셔너리
        """
        # 프롬프트 추출
        prompts = [x["prompt"] for x in inputs]
        
        # 채팅 템플릿 적용
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"] 
            for example in inputs
        ]
        
        # 토크나이징
        prompt_inputs = self.processing_class(
            prompts_text, # 템플릿이 적용된 프롬프트
            return_tensors="pt", # 파이토치 텐서로 반환
            padding=True, # 패딩을 통해 배치 내에서 길이 맞춤
            padding_side="left", # 패딩을 왼쪽에 추가
            add_special_tokens=False # 특별 토큰 추가 안함
        )
        
        # 디바이스로 이동
        prompt_inputs = {k: v.to(device) for k, v in prompt_inputs.items()}
        
        prompt_ids = prompt_inputs["input_ids"] # 프롬프트 토큰 ID
        prompt_mask = prompt_inputs["attention_mask"] # 어텐션 마스크
        
        # 최대 길이 제한 적용
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:] # 최대 길이 제한 적용
            prompt_mask = prompt_mask[:, -self.max_prompt_length:] # 최대 길이 제한 적용
        
        return {
            "prompt_ids": prompt_ids,# 프롬프트 토큰 ID
            "prompt_mask": prompt_mask,# 어텐션 마스크
            "prompts": prompts, # 원본 프롬프트
            "prompts_text": prompts_text # 템플릿이 적용된 프롬프트
        } 