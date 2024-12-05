## Installation
1. Install `requirements.txt`
```
pip install -r requirements.txt
```
2. Install the [evaluation toolkit](https://github.com/MARIO-Math-Reasoning/MARIO_EVAL?tab=readme-ov-file#install-as-python-package) as a package.
3. Install our customized [vllm](https://github.com/MARIO-Math-Reasoning/vllm) to support value model.

Or simply follow the cmds
```bash
> git clone https://github.com/MARIO-Math-Reasoning/Super_MARIO.git
> git clone https://github.com/MARIO-Math-Reasoning/MARIO_EVAL.git
> git clone https://github.com/MARIO-Math-Reasoning/vllm.git

> cd Super_MARIO && pip install -r requirements.txt && cd ..

> cd MARIO_EVAL/latex2sympy && pip install . && cd ..
> pip install -e .

> cd ../vllm
> pip install -e .
```


## Checkpoint Initialization
1. Download the [deepseek-math-7b-base](https://huggingface.co/deepseek-ai/deepseek-math-7b-base).
2. You can use the `scripts/save_value_head.py` to add the value head to the LLM.
3. Maybe we can try qwen-qwq

## Greedy Decoding
You can run either of the following two cmds. There may be slightly difference of accuracy between the two. In our machine, the first got 53.4% and the second got 53.62%.
```
python react_batch_demo.py \
--custom_cfg configs/react_sft.yaml \
--qaf ../MARIO_EVAL/data/math_testset_annotation.json
```
or
```
# use step_beam (1, 1) without value func
python solver_demo.py \
--custom_cfg configs/sbs_greedy.yaml \
--qaf ../MARIO_EVAL/data/math_testset_annotation.json
```


## Step-level Beam Search
In our machine, on MATH testset, the following cmd with config `B1=1, B2=5` can achieve ~62%, and the one with config `B1=3, B2=5` can reach ~65%.
```
python solver_demo.py \
--custom_cfg configs/sbs_sft.yaml \
--qaf ../MARIO_EVAL/data/math_testset_annotation.json
```
Calculate the accuracy
```
python eval_output_jsonl.py \
--res_file <the saved tree jsonl file by solver_demo.py>
```

## MCTS
### Training data generation. 

<img src="imgs/mcts.png">

The `ground_truth` (the final answer, not the solution process) must be provided in `qaf` json or jsonl file (example format can refer to `../MARIO_EVAL/data/math_testset_annotation.json`).

round 1
```
# Checkpoint Initialization is required by adding value head
python solver_demo.py \
--custom_cfg configs/mcts_round1.yaml \
--qaf /path/to/training/data
```

round > 1, after SFT
```
python solver_demo.py \
--custom_cfg configs/mcts_sft_round.yaml \
--qaf /path/to/training/data
```

### Inference. 

Only `question` will be used for solution generation, but the `ground_truth` will be used for calculating the accuracy.
```
python solver_demo.py \
--custom_cfg configs/mcts_sft.yaml \
--qaf ../MARIO_EVAL/data/math_testset_annotation.json
```
Different from step-level beam search, you need first to build a complete tree, then you should run the MCTS offline then calculate the accuracy.
```
python offline_inference.py \
--custom_cfg configs/offline_inference.yaml \
--tree_jsonl <the saved tree jsonl file by solver_demo.py>
```
Note: this evaluation script can also be run with saved tree by step-level beam search, and the accuracy should remain the same.


## Value Estimation

### Distribution of Q-value for intermediate steps on training data. 

Because ground truth is known for training data, the value of final step is reward and Q-value can converge very well.

<img src="imgs/Q_distribution.png" width="500">

### Distribution of Q-value for both intermediate and final steps on test data. 

On test set, the ground truth is unknown, so the Q-value distribution includes both intermediate and final steps. From this figure, we can find
1. When model prediction is correct, its Q-value also converges towards 1.
2. For solutions with incorrect final answer, the distribution of Q-value covers all [-1,1], because the intermediate steps may be correct.
3. When the value model believes the solution predicted by the policy model to be incorrect, the Q-values cluster around $-1$.
4. There are instances where the value model erroneously considers an incorrect solution as correct, where the Q-values have a peak rouand 1. This pattern represents the bad cases of the value model.

<img src="imgs/Q_distribution_test.png" width="500">
