import json
import random
from collections import defaultdict
from functools import lru_cache
from hashlib import md5
from pathlib import Path
from typing import Dict, List, Mapping, Optional, TypedDict

import attrs
import configue
import fire
import jinja2
import jsonlines
from rich.console import Console
from rich.progress import Progress
from rich.table import Table

from scripts.wrapper import AutoTokenizerWrapper

ENVIRONMENT = jinja2.Environment(loader=jinja2.FileSystemLoader("."))


@attrs.define(kw_only=True)
class DatasetConfig:
    input_file: Path = attrs.field(converter=Path)
    output_key: str = "output"
    input_key: Optional[str] = "input"
    instruction_key: Optional[str] = None
    num_samples: Optional[int] = None

    @property
    def identifier(self) -> str:
        return self.input_file.resolve().relative_to(Path("./tasks").resolve()).as_posix()


@attrs.define(kw_only=True)
class DataMixConfig:
    output_dir: Path = attrs.field(converter=Path)
    mix_name: str = "mix"
    datasets: List[DatasetConfig] = attrs.field(factory=list)
    tokenizer: Optional[AutoTokenizerWrapper] = None
    max_length: Optional[int] = None
    validation_split_ratio: float = 0.0
    test_split_ratio: float = 0.0
    random_seed: Optional[int] = 42


class TooLongExampleError(Exception):
    pass


class DatasetStats(TypedDict):
    train: int
    validation: int
    test: int


class DatasetGenerator:
    def __init__(self, datamix_config: DataMixConfig) -> None:
        self.datamix_config = datamix_config
        self._stats: Mapping[str, DatasetStats] = defaultdict(lambda: DatasetStats(train=0, validation=0, test=0))
        random.seed(self.datamix_config.random_seed)

    @staticmethod
    @lru_cache(maxsize=100000)
    def _enforce_word_limit(text: str, max_length: int = 512, tokenizer: Optional[AutoTokenizerWrapper] = None) -> bool:
        """Filter out examples that are too long."""
        count = len(tokenizer(text).input_ids) if tokenizer is not None else len(text.split())
        return count <= max_length

    @staticmethod
    def _compute_hash(text: str) -> int:
        """Compute a hash of the text."""
        return int(md5(text.encode("utf-8")).hexdigest(), 16)

    def _get_split_path(self, split_name: str) -> Path:
        return self.datamix_config.output_dir / f"{self.datamix_config.mix_name}_{split_name}.jsonl"

    def _print_summary(self) -> None:
        table = Table(title="STATS")
        table.add_column("Dataset", style="cyan")
        table.add_column("Train", style="magenta")
        table.add_column("Validation", style="magenta")
        table.add_column("Test", style="magenta")

        for dataset_ref, dataset_stat in self._stats.items():
            table.add_row(
                dataset_ref,
                str(dataset_stat["train"]),
                str(dataset_stat["validation"]),
                str(dataset_stat["test"]),
            )

        console = Console()
        console.print(table)
        total_samples = sum(
            dataset_stat["train"] + dataset_stat["validation"] + dataset_stat["test"]
            for dataset_stat in self._stats.values()
        )
        console.print(f"Total number of samples: {total_samples}")

    def _process_one_dataset_sample(self, config: DatasetConfig, sample: Dict) -> None:
        input_ = ""
        if config.input_key is not None:
            input_ = sample[config.input_key]
        output_ = sample[config.output_key]

        if config.instruction_key is None:
            instruction_template = ENVIRONMENT.get_template(
                (config.input_file.resolve().parent / "instruction.j2").relative_to(Path(".").resolve()).as_posix()
            )
            instruction = instruction_template.render(**sample)
        else:
            instruction = sample[config.instruction_key]

        text_to_write = json.dumps({"instruction": instruction, "input": input_, "output": output_}) + "\n"
        if self.datamix_config.max_length is not None and not self._enforce_word_limit(
            text=text_to_write,
            max_length=self.datamix_config.max_length,
            tokenizer=self.datamix_config.tokenizer,
        ):
            raise TooLongExampleError("word limit not satisfied")

        split = "train"
        if (self.datamix_config.validation_split_ratio + self.datamix_config.test_split_ratio) > 0.0:
            # Deterministically hash the text to decide whether to include it in the train, validation, or test split
            hash = self._compute_hash(text_to_write)
            if hash % 1e9 < (self.datamix_config.validation_split_ratio + self.datamix_config.test_split_ratio) * 1e9:
                split = "validation" if hash % 1e9 < self.datamix_config.validation_split_ratio * 1e9 else "test"

        with open(self._get_split_path(split), "a+", encoding="utf-8") as file:
            file.write(text_to_write)

        self._stats[config.identifier][split] += 1  # type: ignore[literal-required]

    def process(self) -> None:
        self.datamix_config.output_dir.mkdir(parents=True, exist_ok=True)
        for split in ["train", "validation", "test"]:
            self._get_split_path(split_name=split).unlink(missing_ok=True)
        if len(self.datamix_config.datasets) == 0:
            raise ValueError("No datasets specified")

        with Progress() as progress:
            for dataset in progress.track(self.datamix_config.datasets):
                with jsonlines.open(dataset.input_file) as reader:
                    samples = list(reader)
                    random.shuffle(samples)

                progress.console.print(f"Processing {dataset.identifier}")
                ignored_samples = 0
                for sample in samples:
                    if (
                        dataset.num_samples is not None
                        and self._stats[dataset.identifier]["train"] >= dataset.num_samples
                    ):
                        break
                    try:
                        self._process_one_dataset_sample(config=dataset, sample=sample)
                    except TooLongExampleError:
                        ignored_samples += 1
                if ignored_samples > 0:
                    progress.console.print(
                        f"[yellow] {ignored_samples} sample{'s were' if ignored_samples > 1 else ' was'} too long and thus ignored"
                    )
        self._print_summary()


def main(config_file: Path) -> None:
    config: DataMixConfig = configue.load(config_file, sub_path="config")
    DatasetGenerator(config).process()


if __name__ == "__main__":
    fire.Fire(main)
