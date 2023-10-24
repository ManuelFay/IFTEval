# Revisiting Instruction Fine-tuned Model Evaluation to Guide Industrial Applications

This repository contains the code for the EMNLP 2023 paper [Revisiting Instruction Fine-tuned Model Evaluation to Guide Industrial Applications](https://arxiv.org/abs/2310.14103).


## Abstract 

Instruction Fine-Tuning (IFT) is a powerful paradigm that strengthens the zero-shot capabilities of Large Language Models (LLMs), but in doing so induces new evaluation metric requirements. We show LLM-based metrics to be well adapted to these requirements, and leverage them to conduct an investigation of task-specialization strategies, quantifying the trade-offs that emerge in practical industrial settings. Our findings offer practitioners actionable insights for real-world IFT model deployment.

## Citation

If you use this code for your research, please cite our paper:

```
@article{faysse2023revisiting,
  title={Revisiting Instruction Fine-tuned Model Evaluation to Guide Industrial Applications},
  author={Manuel Faysse, Gautier Viaud, Céline Hudelot, Pierre Colombo},
  journal={EMNLP},
  year={2023}
}
```

## Repository structure

This repository contains code, data and experimental results for all experiments in the paper. 
The repository is structured as follows:

### Data

The data folder contains the data used in the paper.
In `data/fractial_mixes`, you will find the various instruction training sets used for both parts of the paper,
evaluation of all scorers (Section 2) and investigations of learning dynamics (Section 3).
Each folder contains data where n samples from the folder name category are included in the training set, along with the 
rest of the synthetic samples from the Alpaca train set. Refer to the paper for more details.
Some task categories are synthetically generated through the Alpaca paradigm, others are manually labeled more classic
NLP tasks (Xsum, SST2, Squad, CONLL) and are used as target tasks in most of the paper's experiments.

The experimental results are added in `data/results`, per experiment as CSV files.
Raw JSON files for all models and experiments are also included in `data/results/results_new.zip`.
To process them for further analysis, a notebook is contained in `scripts/result_aggregation/results_viz.ipynb`.

### Scripts

The `scripts/mix_generation` folder contains all code to sample data mixes from their original datasets.
Note we include only the final sampled datasets in this repo for practicity, and because all datasets used are publicly available on the HuggingFace hub.


The `scripts/result_aggregation` folder contains scripts to convert the raw result files (`data/results/results_new.zip`) with model descriptions
to final result files with each scorer's evaluation, for every experiment.
The final experimental results are added in `data/results`, per experiment as CSV files.
To process them for further analysis, a notebook is contained in `scripts/result_aggregation/results_viz.ipynb`.

### Model training

The `model_training/finetune.py` file contains finetuning results for running instruction finetuning training on 
custom data with Lora adapters, and computing and storing model predictions on the test sets.
It is heavily inspired by tloen's Alpaca-Lora.

SLURM commands used for model training take this form:
```bash
sbatch --job-name=frac_100u --nodes=1 --time=24:00:00 -p gpua100 --gres=gpu:1 --mem-per-cpu=32G --cpus-per-task=8     --output=frac_100u.out     --error=frac_100u.err     --wrap="python model_training/finetune.py       --train_data_path data/fractial_code/fractial_code_100_train.jsonl      --output_dir models/fractial_code_100       --micro_batch_size 32       --num_epochs 2       --cutoff_len 512       --val_data_path data/fractial_code/fractial_code_1000_validation.jsonl       --test_data_path data/fractial_code/fractial_code_only_test.jsonl"
```
and are partially available in the `data/fractial_mixes` folder.

### Evaluation

Evaluation contains all code (called by scripts in `scripts/result_aggregation`) to compute prediction scores for all scorers.

Scorers include:
- Any scorer available through HuggingFace Evaluate
- BertScore
- SentenceBert
- A Reward model trained by OpenAssistant
- API-access LLMs like GPT4 or GPT3.5
- Support for Custom heuristics on custom datasets

### Contact

As this is a repository intended to support reproducing the work in the EMNLP paper, it is not going
to be further developped moving onwards. Some paths to data sources might have been changed since the original code writing.
However, authors will be pleased to answer any implementation detail or questions about the work through Github Issues or email.


### FAQ

The `illuin_llm_tools` package is a proprietary package, which is simply a wrapper around the OpenAI API to 
enable easier parallelization and caching of the API requests.
It is in the process of being open-sourced, in the meantime, simply replacing it's use with classic API calls with the 
OpenAI library yields the same results (lm-evaluator.py).
