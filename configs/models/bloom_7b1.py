from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='bloom_7b1',
        path="bigscience/bloom-7b1",
        tokenizer_path='bigscience/bloom-7b1',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=4,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
