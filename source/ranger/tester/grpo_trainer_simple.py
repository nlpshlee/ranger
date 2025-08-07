from unsloth import FastLanguageModel
from datasets import load_dataset

from ranger.grpo.grpo_config import GRPOConfig
from ranger.grpo.grpo_trainer import GRPOTrainer
# from trl.trainer.grpo_config import GRPOConfig
# from trl.trainer.grpo_trainer import GRPOTrainer


# 모델 로딩 (vllm 비활성화로 안정성 확보)
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "meta-llama/Llama-3.2-1B-Instruct",
    max_seq_length = 512,
    load_in_4bit = True,
    trust_remote_code = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 8,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 8,
    use_gradient_checkpointing = "unsloth",
)

# 3. 더미 reward 함수
def dummy_reward(prompts, completions, answer, **kwargs):
    return [1.0 for _ in completions]

# 4. 샘플 데이터셋
dataset = load_dataset("openai/gsm8k", "main", split="train[:4]").map(lambda x: {
    "prompt": [{"role": "user", "content": x["question"]}],
    "answer": ["42"] * 4
})

# 5. GRPO 설정
config = GRPOConfig(
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 1,
    num_generations = 4,
    max_prompt_length = 128,
    max_completion_length = 256,
    output_dir = "outputs/test",
    max_steps = 1,
    logging_steps = 1,
    report_to = "none"
)

# 6. GRPOTrainer 실행
trainer = GRPOTrainer(
    model = model,
    processing_class = tokenizer,
    reward_funcs = [dummy_reward],
    args = config,
    train_dataset = dataset,
)

trainer.train()