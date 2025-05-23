Run with:

```
CUDA_VISIBLE_DEVICES=0 srun python -m vllm.entrypoints.openai.api_server --model casperhansen/llama-3-70b-instruct-awq --quantization awq --port 5001 --tensor-parallel-size 1 --max-model-len 4096 --disable-log-requests --disable-log-stats &
```

And 

```
CUDA_VISIBLE_DEVICES=0 srun python src/main_evaluate.py --dataset_name wavcaps_test --model_name {model_path} --batch_size 1 --overwrite True --metrics llama3_70b_judge 
```