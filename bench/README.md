
First, get `glaiveai/code_edits_sample` in the trtllm-bench format.
```bash
python make_dataset.py
```

Test at batch size 1:
```bash
trtllm-bench --model nvidia/DeepSeek-V3.1-NVFP4 latency --dataset ./dataset.jsonl --tp 8 --config config.yaml --report_json ./bench_bs_1.json --backend pytorch
```


Test at batch size 8:
```bash
trtllm-bench --model nvidia/DeepSeek-V3.1-NVFP4 latency --dataset ./dataset.jsonl --tp 8 --config config_adp.yaml --report_json ./bench_bs_8.json --backend pytorch
```
