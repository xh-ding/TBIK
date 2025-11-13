import os, sys

import torch
os.environ['VLLM_USE_V1'] = '1'
os.environ["VLLM_ATTENTION_BACKEND"] = "TRITON_ATTN"

from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory

model_path = '/path/to/Qwen3-1.7B'

prompt = ['Please calculate 1+1 step by step.', 'Where is the capital of China?', 'What is your name?', 'How old are you?']
temperature = 0.7
logprobs = 100
generate_tokens = 64
seed=114514

sampling_param = SamplingParams(
        temperature=temperature,
        max_tokens=generate_tokens,
        logprobs=logprobs,
        seed=seed,
)

def vllm_generate(tp_size, batch_size):
    token_id_vllm = []
    text_vllm = []
    logprob_vllm = []

    llm = LLM(model_path, dtype='bfloat16', enforce_eager=True, enable_prefix_caching=False, max_logprobs=logprobs,
              tensor_parallel_size=tp_size, gpu_memory_utilization=0.5, logprobs_mode='raw_logprobs')
    for i in range(len(prompt) // batch_size):
        input_promot = prompt[i * batch_size:min(len(prompt), (i + 1) * batch_size)]
        outputs = llm.generate(input_promot, sampling_param)
        for output in outputs:
            text = output.outputs[0].text
            token_ids = output.outputs[0].token_ids
            logprob = output.outputs[0].logprobs
            converted = [
                [(token_id, lp.logprob) for token_id, lp in step.items()]
                for step in logprob
            ]
            logprob_vllm.append(converted)
            token_id_vllm.append(token_ids)
            text_vllm.append(text)
    del llm
    cleanup_dist_env_and_memory()
    return token_id_vllm, text_vllm, logprob_vllm


if __name__ == '__main__':
    tp_size_1, batch_size_1 = 1, 4
    tp_size_2, batch_size_2 = 4, 2

    batch_invariant_mode = int(os.environ.get('VLLM_BATCH_INVARIANT','0'))
    tp_invariant_mode = int(os.environ.get('VLLM_TP_INVARIANT', '0'))

    _, _, logprob_vllm_1 = vllm_generate(tp_size_1, batch_size_1)
    logprob_vllm_tensor = torch.tensor([[[t[1] for t in seq[:logprobs]] for seq in sample] for sample in logprob_vllm_1], dtype=torch.float32).cuda()
    prob_vllm_tensor = torch.exp(logprob_vllm_tensor)

    _, _, logprob_vllm_2 = vllm_generate(tp_size_2, batch_size_2)
    logprob_vllm_tensor2 = torch.tensor([[[t[1] for t in seq[:logprobs]] for seq in sample] for sample in logprob_vllm_2], dtype=torch.float32).cuda()
    prob_vllm_tensor2 = torch.exp(logprob_vllm_tensor2)
    print(f'Batch-Invariant Mode {tp_invariant_mode}')
    print(f'TP-Invariant Mode {tp_invariant_mode}')
    print(f'Setting 1: tp_size={tp_size_1}, batch_size={batch_size_1}')
    print(f'Setting 2: tp_size={tp_size_2}, batch_size={batch_size_2}')
    print('-'*50)
    print(f'\tMax Difference Probability: {(prob_vllm_tensor2-prob_vllm_tensor).abs().max().item()*100}%')