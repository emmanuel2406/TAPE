# Expert Parallelism Load Balancer (EPLB)

This example demonstrates how to use OpenEvolve to optimize the Expert Parallelism Load Balancer (EPLB) algorithm.
**Previous work to be reproduced**

## Setup

Install PyTorch:

```bash
uv pip install torch
```

Download the workload file from [Hugging Face](https://huggingface.co/datasets/abmfy/eplb-openevolve):

```bash
wget https://huggingface.co/datasets/abmfy/eplb-openevolve/resolve/main/expert-load.json
```

The original evaluator from Barbarians at the Gate is in `evaluator.py`. 

Our new evaluator which also simulates the communication between GPUs during inference is in `TAPE_evaluator.py`.