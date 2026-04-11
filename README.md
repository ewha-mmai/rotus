# Rotus: Calibrated Refusal Optimization for Multimodal Tool-Use Agents
Hyemin Boo, Yeongeun Byeon, Jiyoung Lee

![intro](asset/intro.jpg)
### Abstract
> Agentic multimodal systems have shown impressive performance with multi-turn reasoning and tool invocation for complex vision-language tasks.
> However, increased execution capability does not necessarily translate into reliable decision-making.
> Our interesting findings reveal that SOTA agents, finetuned on RL from multimodal large language models (MLLMs), often fail to abstain in unanswerable scenarios, while degrading the refusal behavior in the base MLLMs.
> In this work, we propose ***Rotus*** to calibrate agents' refusal behavior through group relative policy optimization (GRPO) by the accuracy reward with a turn-dependent penalty.
> Our reward formulation jointly models answer correctness, calibrated refusal, and reasoning efficiency while preserving tool-use capability.
> We further introduce ***HARDBench***, a benchmark with systematically constructed unanswerable questions in high-complexity visual scenarios, including charts, counting tasks, and real-world reasoning problems.
> Extensive experiments show that ***Rotus*** improves refusal reliability while maintaining, and in some cases improving, answer accuracy with only a lightweight calibration stage.

# Environment

## Training
Training runs inside Docker — no separate environment setup needed. See the [Docker Setup](#docker-setup) section below.

## Judge Server & Inference

```bash
conda create -n vllm python=3.10
conda activate vllm
pip install vllm==0.19.0
pip install openai pillow tqdm requests
```

# Data

## HARDBench

HARDBench is our benchmark with systematically constructed unanswerable questions in high-complexity visual scenarios.
The dataset is available at [HARDBench](https://huggingface.co/datasets/hyeminboo/HARDBench).

## Image Data

HARDBench questions are derived from the following source datasets. Download images from each source:

| Dataset | Link |
|---|---|
| ArxivQA | [MMInstruction/ArxivQA](https://huggingface.co/datasets/MMInstruction/ArxivQA) |
| MM-adaptive-CoF RL | [xintongzhang/CoF-RL-Data](https://huggingface.co/datasets/xintongzhang/CoF-RL-Data) |
| PixMo-Count | [allenai/pixmo-count](https://huggingface.co/datasets/allenai/pixmo-count) |
| TallyQA | [tallyqa.zip](https://github.com/manoja328/tallyqa/blob/master/tallyqa.zip?raw=true) |
| RealXBench | [glowol/RealXBench](https://huggingface.co/datasets/glowol/RealXBench) |
| V*Bench | [craigwu/vstar_bench](https://huggingface.co/datasets/craigwu/vstar_bench) |

## Data Conversion

To convert HARDBench into VeRL training format for each model:

```bash
# CoF
python scripts/convert_cof.py \
    --input HARDBench_train.json \
    --output train_cof.json \
    --image_root /workspace/data/Dataset

# DeepEyesV2
python scripts/convert_deepeyesv2.py \
    --input HARDBench_train.json \
    --output train_deepeyesv2.json \
    --image_root /workspace/data/Dataset

# Thyme
python scripts/convert_thyme.py \
    --input HARDBench_train.json \
    --output train_thyme.json \
    --image_root /workspace/data/Dataset \
    --local_image_root /path/to/local/Dataset
```

# Training
![framework](asset/framework.jpg)
## Model Weights

Download and store all pretrained weights under `/models/`:

| Model | HuggingFace |
|---|---|
| CoF-rl-model-7b (base) | [xintongzhang/CoF-rl-model-7b](https://huggingface.co/xintongzhang/CoF-rl-model-7b) |
| DeepEyesV2_7B_1031 (base) | [honglyhly/DeepEyesV2_7B_1031](https://huggingface.co/honglyhly/DeepEyesV2_7B_1031) |
| Thyme-RL (base) | [Kwai-Keye/Thyme-RL](https://huggingface.co/Kwai-Keye/Thyme-RL) |
| Rotus-CoF | coming soon |
| Rotus-DeepEyesV2 | coming soon |
| Rotus-Thyme | coming soon |

```
/models/
  ├── Chain-of-Focus/
  │   └── CoF-rl-model-7b/          ← base model
  ├── Rotus-CoF/                  ← Rotus-trained weights
  ├── DeepEyesV2/
  │   └── DeepEyesV2_7B_1031/       ← base model
  ├── Rotus-DeepEyesV2/           ← Rotus-trained weights
  ├── Thyme/
  │   └── Thyme-RL/                  ← base model
  ├── Rotus-Thyme/                   ← Rotus-trained weights
  └── Qwen2.5-72B-Instruct-AWQ/     ← judge server
```

Training requires the following components to be running before starting:

| Component | Required for |
|---|---|
| Judge Server | All models |
| Sandbox Server | DeepEyesV2 only |
| Training container (Docker) | All models |

## Step 1. Judge Server

Run on the host machine (outside Docker) using the `vllm` conda environment:

```bash
conda activate vllm

CUDA_VISIBLE_DEVICES=0 vllm serve /path/to/models/Qwen2.5-72B-Instruct-AWQ \
    --port 18901 \
    --dtype float16 \
    --gpu-memory-utilization 0.95 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --served-model-name "judge" \
    --trust-remote-code \
    --quantization awq_marlin
```

## Step 2. Sandbox Server (DeepEyesV2 only)

```bash
# Stop and remove existing container
docker stop deepeyes-v2 && docker rm deepeyes-v2

docker run -d \
  -v /path/to/data:/data1 \
  -p 28901-28904:18901-18904 \
  --add-host=host.docker.internal:172.17.0.1 \
  --name deepeyes-v2 \
  chenshawn6915/multimodal-ipython-sandbox:oss-v2

# Verify
docker ps | grep deepeyes-v2
curl -X POST http://127.0.0.1:8000/run_jupyter \
  -H "Content-Type: application/json" \
  -d '{"session_id": "test123", "code": "print(\"hello\")", "timeout": 10}' \
  --max-time 5
```

The sandbox container requires a one-time patch to fix a session state persistence bug in `local_jupyter_session.py`.
Without this patch, code execution results will not persist across turns in multi-turn training.

```bash
# Copy patch script into container and apply
docker cp scripts/patch_jupyter.py deepeyes-v2:/tmp/patch_jupyter.py
docker exec deepeyes-v2 python /tmp/patch_jupyter.py

# Verify
curl -s -X POST http://127.0.0.1:8000/run_jupyter \
  -H "Content-Type: application/json" \
  -d '{"session_id": "check1", "code": "print(1)", "timeout": 5}'
```

Alternatively, use the provided script (skips patch if already applied):

```bash
HOST_PATCH_PATH=scripts/patch_jupyter.py bash scripts/repatch_jupyter.sh
```

## Step 3. Docker Setup

```bash
# Remove existing container
docker rm -f verl

# Create container
docker run -dit \
  --runtime=nvidia \
  --gpus all \
  --net=host \
  --shm-size="100g" \
  --cap-add=SYS_ADMIN \
  -v /path/to/train/verl:/workspace/verl \
  -v /path/to/data:/workspace/data \
  -v /path/to/models:/workspace/models \
  -v /path/to/inference:/workspace/inference \
  -v /path/to/ray_tmp:/workspace/ray_tmp \
  --name verl \
  --entrypoint /bin/bash \
  verlai/verl:vllm011.latest

# Access container
docker exec -it verl bash

# Install verl
cd /workspace/verl
pip install --no-deps -e .
python -c "import verl; print('OK')"
```

## Step 4. Training Script

Run inside the Docker container:

```bash
# CoF
bash examples/grpo_trainer/Rotus/run_cof.sh

# DeepEyesV2
bash examples/grpo_trainer/Rotus/run_deepeyesv2.sh

# Thyme
bash examples/grpo_trainer/Rotus/run_thyme.sh
```



# Evaluation

## Step 1. Merge FSDP Checkpoint

After training, merge the FSDP checkpoint to HuggingFace format before serving.
Run inside the Docker container:

```bash
cd /workspace/verl
python -m verl.model_merger merge \
    --backend fsdp \
    --local_dir /path/to/checkpoints/global_step_N/actor \
    --target_dir /path/to/checkpoints/merged_hf
```

## Step 2. Run Inference

Inference is parallelized across multiple workers, using the `vllm` conda environment:

```bash
# CoF
CUDA_VISIBLE_DEVICES=0 nohup python -m vllm.entrypoints.openai.api_server \
    --model /path/to/merged_hf --port 8001 --trust-remote-code \
    --max-model-len 20000 --gpu-memory-utilization 0.9 \
    --limit-mm-per-prompt '{"image": 10}' &

python eval/CoF/inference_cof.py \
    --data_path /path/to/test.json \
    --image_base_dir /path/to/Dataset \
    --output_path results_cof.jsonl \
    --api_base http://localhost:8001/v1 \
    --max_turns 5
```

```bash
# DeepEyesV2 (sandbox server must be running)
python eval/DeepEyesV2/inference_deepeyesv2.py \
    --data_path /path/to/test.json \
    --image_base_dir /path/to/Dataset \
    --output_path results_deepeyesv2.jsonl \
    --api_base http://localhost:8001/v1 \
    --code_sandbox_url http://127.0.0.1:8000/run_jupyter \
    --max_turns 5
```

```bash
# Thyme
python eval/Thyme/inference_thyme.py \
    --data_path /path/to/test.json \
    --image_base_dir /path/to/Dataset \
    --output_path results_thyme.jsonl \
    --api_base http://localhost:8001/v1 \
    --thyme_path /path/to/models/Thyme \
    --max_turns 5
```

After inference, shut down the vLLM server:
```bash
pkill -f vllm.entrypoints.openai.api_server
```

## Step 3. Evaluate

Evaluation uses GPT as a judge. Set your OpenAI API key before running.

**Accuracy & Refusal** (`evaluate_results.py`):
```bash
# Submit batch to GPT
python eval/evaluate_results.py submit \
    --input_path results_cof.jsonl \
    --out_dir eval_output/ \
    --api_key $OPENAI_API_KEY

# Collect results after batch completes
python eval/evaluate_results.py collect \
    --input_path results_cof.jsonl \
    --out_dir eval_output/ \
    --api_key $OPENAI_API_KEY \
    --batch_id <batch_id>
```

**Refusal Quality** (`evaluate_rationality.py`):
```bash
python eval/evaluate_rationality.py \
    --input_path results_cof.jsonl \
    --output_path rationality_cof.json \
    --api_key $OPENAI_API_KEY
```


# Acknowledgement

We thank the following projects for their open-source contributions, which this work builds upon: [Chain-of-Focus](https://github.com/xtong-zhang/Chain-of-Focus), [Thyme](https://github.com/yfzhang114/Thyme), [DeepEyesV2](https://github.com/Visual-Agent/DeepEyesV2), [VeRL](https://github.com/verl-project/verl)

# Citation

If you find this work useful, please consider citing:

```bibtex
coming soon...
```
