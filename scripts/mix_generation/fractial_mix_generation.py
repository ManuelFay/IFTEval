import os
from pathlib import Path

import fire
from transformers import AutoTokenizer

from scripts.generate_data_mix import (DataMixConfig, DatasetConfig,
                                       DatasetGenerator)

categories = [
    "Classify",
    "Code",
    "ContextAnswer",
    "Create",
    "Extract",
    "Logic",
    "MemoryAnswer",
    "MemorySummarize",
    "Rewrite",
    "Summarize",
    "Translate",
    "Write",
]

DATA_PATH = Path("./tasks/self_instruct/original_alpaca/categories")
OUTPUT_PATH = Path("./outputs/fractial_mixes/")
SLURM_COMMAND = """
sbatch --job-name=frac_{n}u --nodes=1 --time=24:00:00 -p gpua100 --gres=gpu:1 --mem-per-cpu=32G --cpus-per-task=8 \
    --output=frac_{n}u.out \
    --error=frac_{n}u.err \
    --wrap="python model_training/finetune.py \
      --train_data_path data/fractial_{cat}/fractial_{cat}_{n}_train.jsonl\
      --output_dir models/fractial_{cat}_{n} \
      --micro_batch_size 32 \
      --num_epochs 2 \
      --cutoff_len 512 \
      --val_data_path data/fractial_{cat}/fractial_{cat}_1000_validation.jsonl \
      --test_data_path data/fractial_{cat}/fractial_{cat}_only_test.jsonl"
"""


def main() -> None:
    tokenizer = AutoTokenizer.from_pretrained("illuin/french-alpaca-v1", use_auth_token=True, use_fast=False)

    if not os.path.exists(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    # if os.path.exists(f"{OUTPUT_PATH}/slurm_launch.sh"):
    #     os.remove(f"{OUTPUT_PATH}/slurm_launch.sh")

    for cat in categories:
        config = DataMixConfig(
            mix_name=f"fractial_{cat.lower()}_only",
            output_dir=f"{OUTPUT_PATH}/fractial_{cat.lower()}",
            tokenizer=tokenizer,
            max_length=512,
            validation_split_ratio=0.1,
            test_split_ratio=0.1,
            random_seed=42,
            datasets=[
                DatasetConfig(
                    input_file=f"{DATA_PATH}/{cat}/alpaca_gpt4_en.jsonl",
                    instruction_key="instruction",
                    num_samples=1300,
                )
            ],
        )
        DatasetGenerator(config).process()

        for n in [0, 10, 100, 1000]:
            datasets = []
            for cat2 in categories:
                datasets.append(
                    DatasetConfig(
                        input_file=f"{DATA_PATH}/{cat2}/alpaca_gpt4_en.jsonl",
                        instruction_key="instruction",
                        num_samples=None if cat2 != cat else n,
                    )
                )

            mix_name = f"fractial_{cat.lower()}_{n}"
            output_dir = f"{OUTPUT_PATH}/fractial_{cat.lower()}"

            config = DataMixConfig(
                mix_name=mix_name,
                output_dir=output_dir,
                tokenizer=tokenizer,
                max_length=512,
                validation_split_ratio=0.1,
                test_split_ratio=0.1,
                random_seed=42,
                datasets=datasets,
            )
            DatasetGenerator(config).process()

            with open(f"{OUTPUT_PATH}/slurm_launch.sh", "a+") as f:
                f.write(SLURM_COMMAND.format(n=n, cat=cat.lower()))


if __name__ == "__main__":
    fire.Fire(main)
