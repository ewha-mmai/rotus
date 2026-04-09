# Training

## Docker Setup

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

## Judge Server

```bash
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

## DeepEyesV2 Sandbox Server

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
Without this patch, code execution results will not persist across turns in multi-turn inference.

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

## Training Script

```bash
# Script varies by model
bash examples/grpo_trainer/run_deepeyes_mohobench.sh
```
