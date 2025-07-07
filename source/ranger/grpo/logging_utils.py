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

from typing import Optional
from packaging import version
import transformers


def log_metrics(logs: dict[str, float], metrics: dict[str, list], 
                super_log_func=None, start_time: Optional[float] = None) -> None:
    """
    메트릭을 계산하고 로깅하는 함수
    
    Args:
        logs: 로깅할 로그 데이터
        metrics: 메트릭 데이터 (키별로 값들의 리스트)
        super_log_func: 부모 클래스의 log 함수
        start_time: 시작 시간
    
    Returns:
        None
    """
    # 메트릭 평균 계산
    calculated_metrics = {key: sum(val) / len(val) for key, val in metrics.items()}

    # 이 메서드는 훈련과 평가 모두에서 호출될 수 있습니다. 평가에서 호출될 때 `logs`의 키들은
    # "eval_"로 시작합니다. 형식을 맞추기 위해 `metrics`의 키들에 "eval_" 접두사를 추가해야 합니다.
    if next(iter(logs.keys())).startswith("eval_"):
        calculated_metrics = {f"eval_{key}": val for key, val in calculated_metrics.items()}

    logs = {**logs, **calculated_metrics}
    
    if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
        super_log_func(logs, start_time)
    else:  # transformers<=4.46
        super_log_func(logs)
    
    # 메트릭 초기화
    metrics.clear() 