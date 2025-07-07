from unsloth import FastLanguageModel
import torch
max_seq_length = 2048 # 더 긴 추론 과정을 위해 증가 가능
lora_rank = 16 # 랭크가 클수록 더 똑똑하지만 느림

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/meta-Llama-3.1-8B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = False, # LoRA 16비트를 위해 False로 설정
    fast_inference = True, # vLLM 빠른 추론 활성화
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.6, # 메모리 부족시 감소
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # 0보다 큰 숫자 선택! 추천값: 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ], # 메모리 부족시 QKVO 제거
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # 긴 컨텍스트 파인튜닝 활성화
    random_state = 3407,
)
# datasets 라이브러리에서 load_dataset 함수를 가져옵니다
from datasets import load_dataset

# OpenAI의 GSM8K 데이터셋을 로드합니다
# "main" 스플릿의 "train" 데이터를 가져옵니다
dataset = load_dataset("openai/gsm8k", "main", split = "train")

def extract_hash_answer(text):
    # 텍스트에 "####"가 없으면 None 반환
    if "####" not in text: return None
    # "####" 이후의 텍스트를 추출하고 앞뒤 공백 제거
    return text.split("####")[1].strip()
# 데이터셋의 첫 번째 답변에서 해시 태그 이후의 답변 추출
extract_hash_answer(dataset[0]["answer"])


# 추론 시작과 끝을 나타내는 태그 정의
reasoning_start = "<start_working_out>"
reasoning_end   = "<end_working_out>"
# 해답 시작과 끝을 나타내는 태그 정의 
solution_start = "<SOLUTION>"
solution_end = "</SOLUTION>"

# 시스템 프롬프트 템플릿 정의
system_prompt = \
f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {reasoning_start} and {reasoning_end}.
Then, provide your solution between {solution_start}{solution_end}"""

# 데이터셋을 매핑하여 각 샘플에 대해 다음 작업 수행:
# 1. 시스템 프롬프트와 사용자 질문으로 구성된 프롬프트 생성
# 2. 답변에서 해시 태그 이후의 정답 추출
dataset = dataset.map(lambda x: {
    "prompt" : [
        {"role": "system", "content": system_prompt},  # 시스템 프롬프트 설정
        {"role": "user",   "content": x["question"]},  # 사용자 질문 추가
    ],
    "answer": extract_hash_answer(x["answer"]),  # 답변에서 해시 태그 이후의 정답 추출
})

import re

# 정규표현식 패턴 정의
# ^[\s]{0,} : 문자열 시작 부분의 공백 문자 0개 이상 매칭
# {reasoning_start}.+?{reasoning_end} : 추론 시작/끝 태그 사이의 내용 매칭
# .*? : 추론과 해답 사이의 내용 매칭
# {solution_start}(.+?){solution_end} : 해답 시작/끝 태그 사이의 내용을 그룹으로 캡처
# [\s]{0,}$ : 문자열 끝 부분의 공백 문자 0개 이상 매칭
# MULTILINE : 여러 줄 매칭 허용
# DOTALL : 개행 문자를 포함한 모든 문자 매칭
match_format = re.compile(
    rf"^[\s]{{0,}}"\
    rf"{reasoning_start}.+?{reasoning_end}.*?"\
    rf"{solution_start}(.+?){solution_end}"\
    rf"[\s]{{0,}}$",
    flags = re.MULTILINE | re.DOTALL
)

# reward 함수 설계
def match_format_exactly(completions, **kwargs):
    scores = []  # 점수를 저장할 리스트 초기화
    for completion in completions:
        score = 0  # 각 완성문에 대한 점수 초기화
        response = completion[0]["content"]  # 완성문의 내용 추출
        # 정확한 형식이 일치하면 3점 부여
        if match_format.search(response) is not None: score += 3.0
        scores.append(score)  # 계산된 점수를 리스트에 추가
    return scores  # 모든 점수 반환

def match_format_approximately(completions, **kwargs):
    scores = []  # 점수를 저장할 리스트 초기화
    for completion in completions:
        score = 0  # 각 완성문에 대한 점수 초기화
        response = completion[0]["content"]  # 완성문의 내용 추출
        
        # 각 키워드가 정확히 1번씩만 나타나는지 확인
        # 1번만 나타나면 0.5점 부여, 그렇지 않으면 -1.0점 부여
        score += 0.5 if response.count(reasoning_start) == 1 else -1.0  # 추론 시작 태그
        score += 0.5 if response.count(reasoning_end)   == 1 else -1.0  # 추론 종료 태그
        score += 0.5 if response.count(solution_start)  == 1 else -1.0  # 해답 시작 태그
        score += 0.5 if response.count(solution_end)    == 1 else -1.0  # 해답 종료 태그
        
        scores.append(score)  # 계산된 점수를 리스트에 추가
    return scores  # 모든 점수 반환

def check_answer(prompts, completions, answer, **kwargs):
    # 질문과 응답 추출
    question = prompts[0][-1]["content"]
    responses = [completion[0]["content"] for completion in completions]

    # 정규식 패턴을 사용하여 응답에서 답변 추출
    extracted_responses = [
        guess.group(1)
        if (guess := match_format.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # 각 추출된 답변과 실제 답변을 비교하여 점수 계산
    for guess, true_answer in zip(extracted_responses, answer):
        score = 0
        # 답변이 추출되지 않은 경우 0점
        if guess is None:
            scores.append(0)
            continue
        # 정확한 답변일 경우 3점 부여
        if guess == true_answer:
            score += 3.0
        # 공백만 다른 경우 1.5점 부여
        elif guess.strip() == true_answer.strip():
            score += 1.5
        else:
            # 답변이 근접한 경우 비율에 따라 점수 부여
            try:
                ratio = float(guess) / float(true_answer)
                # 90%~110% 범위: 1점
                if   ratio >= 0.9 and ratio <= 1.1: score += 1.0
                # 80%~120% 범위: 0.5점
                elif ratio >= 0.8 and ratio <= 1.2: score += 0.5
                else: score -= 1.5 # 잘못된 답변은 1.5점 감점
            except:
                score -= 1.5 # 숫자 변환 실패 시 1.5점 감점
        scores.append(score)
    return scores

# 숫자, 소수점, 쉼표를 포함하는 패턴을 찾는 정규식 컴파일
match_numbers = re.compile(
    solution_start + r".*?([\d\.\,]{1,})",  # solution_start 태그 다음에 오는 숫자, 소수점, 쉼표 패턴 매칭
    flags = re.MULTILINE | re.DOTALL  # 여러 줄 매칭과 개행 문자 포함 매칭 활성화
)

# 전역 변수 초기화
global PRINTED_TIMES
PRINTED_TIMES = 0
global PRINT_EVERY_STEPS
PRINT_EVERY_STEPS = 5

def check_numbers(prompts, completions, answer, **kwargs):
    # 프롬프트에서 질문 추출
    question = prompts[0][-1]["content"]
    # 완성된 응답들 추출
    responses = [completion[0]["content"] for completion in completions]

    # 정규식을 사용하여 숫자 추출
    extracted_responses = [
        guess.group(1)
        if (guess := match_numbers.search(r)) is not None else None \
        for r in responses
    ]

    scores = []
    # 일정 간격으로만 출력하도록 설정
    global PRINTED_TIMES
    global PRINT_EVERY_STEPS
    if PRINTED_TIMES % PRINT_EVERY_STEPS == 0:
        print('*'*20, f"Question:\n{question}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}", f"\nExtracted:\n{extracted_responses[0]}")
    PRINTED_TIMES += 1

    # 추출된 답변과 실제 답변 비교
    for guess, true_answer in zip(extracted_responses, answer):
        if guess is None:
            scores.append(0)
            continue
        # 숫자로 변환
        try:
            true_answer = float(true_answer.strip())
            # 쉼표 제거 (예: 123,456)
            guess       = float(guess.strip().replace(",", ""))
            # 정확한 답변은 1.5점, 틀린 답변은 -0.5점
            scores.append(1.5 if guess == true_answer else -0.5)
        except:
            scores.append(0)
            continue
    return scores

max_prompt_length = 287 + 1 # 안전을 위해 1을 더함

from trl import GRPOConfig, GRPOTrainer
training_args = GRPOConfig(
    learning_rate = 5e-6, # 학습률 설정
    weight_decay = 0.1, # 가중치 감쇠 설정
    warmup_ratio = 0.1, # 워밍업 비율 설정
    lr_scheduler_type = "cosine", # 코사인 스케줄러 사용
    optim = "adamw_8bit", # 8비트 AdamW 옵티마이저 사용
    logging_steps = 1, # 로깅 간격 설정
    per_device_train_batch_size = 8, # 디바이스당 배치 크기
    gradient_accumulation_steps = 1, # 더 부드러운 학습을 위해 4로 증가
    num_generations = 4, # 메모리 부족시 감소 가능
    max_prompt_length = max_prompt_length, # 최대 프롬프트 길이
    max_completion_length = max_seq_length - max_prompt_length, # 최대 완성 길이
    # num_train_epochs = 1, # 전체 학습을 위해 1로 설정
    max_steps = 500, # 최대 학습 스텝
    save_steps = 250, # 모델 저장 간격
    max_grad_norm = 1.0, # 그래디언트 클리핑
    report_to = "none", # Weights & Biases 사용 가능
    output_dir = "outputs", # 출력 디렉토리
)

# GRPO 트레이너 초기화 및 학습 실행
trainer = GRPOTrainer(
    model = model,  # 학습할 모델
    processing_class = tokenizer,  # 토크나이저
    reward_funcs = [  # 보상 함수들
        match_format_exactly,  # 정확한 형식 매칭
        match_format_approximately,  # 근사 형식 매칭
        check_answer,  # 답변 검증
        check_numbers,  # 숫자 검증
    ],
    args = training_args,  # 학습 설정
    train_dataset = dataset,  # 학습 데이터셋
)
trainer.train()  # 학습 시작

# # 사용자 질문을 채팅 템플릿에 적용
# text = tokenizer.apply_chat_template([
#     {"role": "user", "content": "What is the sqrt of 101?"},
# ], tokenize = False, add_generation_prompt = True)

# # vllm에서 샘플링 파라미터 임포트
# from vllm import SamplingParams
# # 텍스트 생성에 사용할 샘플링 파라미터 설정
# sampling_params = SamplingParams(
#     temperature = 0.8,  # 생성 다양성 조절 (높을수록 더 다양)
#     top_p = 0.95,      # 누적 확률 임계값
#     max_tokens = 1024, # 최대 토큰 수 제한
# )
# # 모델을 사용하여 텍스트 생성
# output = model.fast_generate(
#     [text],
#     sampling_params = sampling_params,
#     lora_request = None,  # LoRA 가중치 사용하지 않음
# )[0].outputs[0].text

# print("기존 학습하지 않은 모델")
# print(output)
# model.save_lora("grpo_saved_lora")
# from safetensors import safe_open

# tensors = {}
# with safe_open("grpo_saved_lora/adapter_model.safetensors", framework = "pt") as f:
#     # A와 B 텐서가 모두 0이 아닌지 확인
#     for key in f.keys():
#         tensor = f.get_tensor(key)
#         n_zeros = (tensor == 0).sum() / tensor.numel()
#         assert(n_zeros.item() != tensor.numel())
# # 시스템 프롬프트와 사용자 질문을 포함하는 메시지 리스트 생성
# messages = [
#     {"role": "system", "content": system_prompt},
#     {"role": "user",   "content": "What is the sqrt of 101?"},
# ]

# # 토크나이저를 사용하여 채팅 템플릿 적용
# text = tokenizer.apply_chat_template(
#     messages,
#     add_generation_prompt = True, # 생성 시 반드시 필요
#     tokenize = False,
# )

# # VLLM에서 샘플링 파라미터 임포트
# from vllm import SamplingParams

# # 텍스트 생성에 사용할 샘플링 파라미터 설정
# sampling_params = SamplingParams(
#     temperature = 0.8,  # 생성 다양성 조절 (높을수록 더 다양)
#     top_p = 0.95,      # 누적 확률 임계값
#     max_tokens = 1024, # 최대 토큰 수 제한
# )

# # LoRA 가중치를 로드하여 모델로 텍스트 생성
# output = model.fast_generate(
#     text,
#     sampling_params = sampling_params,
#     lora_request = model.load_lora("grpo_saved_lora"), # 학습된 LoRA 가중치 로드
# )[0].outputs[0].text
# print("GRPO적용")
# print(output)