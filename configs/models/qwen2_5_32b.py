from opencompass.models import HuggingFaceCausalLM
import torch

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen2.5_32B',
        path="Qwen/Qwen2.5-32B",
        tokenizer_path="Qwen/Qwen2.5-32B",
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=2,
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
