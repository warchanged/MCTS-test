from __future__ import annotations

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import argparse

from omegaconf import OmegaConf
from termcolor import colored
from tqdm import tqdm

from mcts_math.agents import SBSREACT
#from mcts_math.agents import MCTS
from mcts_math.solver import Solver
from mcts_math.config import BaseConfig
from react_demo import load_qaf
from react_batch_demo import batch


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('--custom_cfg', type=str, default="configs/sbs_sft.yaml")
    args.add_argument(
        "--qaf", "--question-answer-file", 
        type=str, 
        required=True,
        help="the file includes quesiton / partial solution (optional) / answer (optional)")

    args = args.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    config = OmegaConf.structured(BaseConfig)
    if args.custom_cfg:
        custom_config = OmegaConf.load(args.custom_cfg)
        config = OmegaConf.merge(config, custom_config)
    config = OmegaConf.create(OmegaConf.to_yaml(config, resolve=True))
    print(config)

    llm_version = os.path.basename(config.model_dir.rstrip("/"))

    data = load_qaf(args.qaf)
    solver = Solver(config=config)

    # init method
    #if config.mode == "mcts":
    #    method = MCTS
    if config.mode == "sbs":
        method = SBSREACT
    else:
        raise NotImplementedError

    saved_jsonl_file = f"{args.qaf}.{config.mode}.{llm_version}.jsonl" 
    with open(saved_jsonl_file, "w") as writer:
        for cur_data in tqdm(batch(data, config.batch_size), desc="Main Processing"):
            agents = [method(config=config, question=d["question"]) for d in cur_data]
            jsonlines = solver.solve(agents)
            for d in cur_data:
                question = d["question"]
                d["react"] = jsonlines[question]
                writer.write(json.dumps(d, ensure_ascii=False) + '\n')
                writer.flush()
