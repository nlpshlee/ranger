# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Optional


def prediction_step(model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None, 
                   prepare_inputs_func=None, compute_loss_func=None, compute_loss_context_manager=None):
    """
    평가(evaluation) 단계에서 모델의 성능을 측정하는 함수
    
    Args:
        model: 평가할 모델
        inputs: 모델 입력 데이터
        prediction_loss_only: 손실만 예측할지 여부
        ignore_keys: 무시할 키들
        prepare_inputs_func: 입력 데이터를 준비하는 함수
        compute_loss_func: 손실을 계산하는 함수
        compute_loss_context_manager: 손실 계산 컨텍스트 매니저
    
    Returns:
        tuple: (loss, predictions, labels) - 손실, 예측값, 레이블
    """
    inputs = prepare_inputs_func(inputs)
    with torch.no_grad():
        with compute_loss_context_manager():
            loss = compute_loss_func(model, inputs)
        loss = loss.mean().detach()
    return loss, None, None 