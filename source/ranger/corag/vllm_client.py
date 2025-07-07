import torch
from typing import List, Dict, Union
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from openai.types.chat import ChatCompletion
from vllm import LLM, SamplingParams
# from src.utils import AtomicCounter


def get_vllm_model_id(host: str = "localhost", port: int = 8000, api_key: str = "token-123") -> str:
    openai_api_base = f"http://{host}:{port}/v1"

    client = OpenAI(
        api_key=api_key,
        base_url=openai_api_base,
    )

    models = client.models.list()
    return models.data[0].id


class VllmClient:
    def __init__(self, model: str, host: str = 'localhost', port: int = 8000, api_key: str = 'token-123'):
        super().__init__()
        self.model = model
        self.client: OpenAI = OpenAI(
            base_url=f"http://{host}:{port}/v1",
            api_key=api_key,
        )
        # 메모리 사용량 최적화를 위한 LLM 초기화 옵션
        # self.llm = LLM(
        #     model=model,
        #     dtype="half",
        # )
        from ranger.corag.utils import AtomicCounter
        self.token_consumed: AtomicCounter = AtomicCounter()
        self.called_cnt = 0

    def call_chat(self, messages: List[Dict], return_str: bool = True, **kwargs) -> Union[str, ChatCompletion]:
        if not messages:
            return ""
        
        completion: ChatCompletion = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )
        self.token_consumed.increment(num=completion.usage.total_tokens)

        self.called_cnt += 1
        if self.called_cnt % 1000 == 0:
            print(f'VllmClient.call_chat() called_cnt : {self.called_cnt}')
        
        return completion.choices[0].message.content if return_str else completion

    def batch_call_chat(self, messages: List[List[Dict]], return_str: bool = True, num_workers: int = 4, **kwargs) -> List[Union[str, ChatCompletion]]:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            return list(executor.map(
                lambda m: self.call_chat(m, return_str=return_str, **kwargs),
                messages
            ))
            

    # def chatml_prompt(self, messages, system_prompt=None):
    #     prompt = "<|begin_of_text|>\n"
    #     if system_prompt:
    #         prompt += "<|start_header_id|>system<|end_header_id|>\n"
    #         prompt += system_prompt.strip() + "\n"
    #     for msg in messages:
    #         role = msg["role"]
    #         content = msg["content"].strip()
    #         prompt += f"<|start_header_id|>{role}<|end_header_id|>\n{content}\n"
    #     prompt += "<|start_header_id|>assistant<|end_header_id|>\n"
    #     return prompt

    # def new_call_chat(self, messages: List[List[Dict]], temperature=0.7, max_tokens=4096):
    #     for x in messages:
    #         if len(x) > 1:
    #             print("prompt 예외 발생, new_call_chat 함수 확인 필요")
    #     print(self.chatml_prompt(messages[0]))
    #     messages = [self.chatml_prompt(x) for x in messages]
    #     # print(messages)
    #     sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    #     outputs = self.llm.generate(messages, sampling_params, use_tqdm=True)
        
    #     result = []
    #     for output in outputs:
    #         text = output.outputs[0].text.strip()
    #         result.append(text)

    #     return result
    # def new_call_chat_2(self, messages: List[List[Dict]], temperature=0.7, max_tokens=4096):
    #     for x in messages:
    #         if len(x) > 1:
    #             print("prompt 예외 발생, new_call_chat 함수 확인 필요")
    #     messages = [x[0] for x in messages]
    #     # print(messages)
    #     sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    #     outputs = self.llm.chat(messages, sampling_params, use_tqdm=True)
        
    #     result = []
    #     for output in outputs:
    #         text = output.outputs[0].text.strip()
    #         result.append(text)

    #     return result