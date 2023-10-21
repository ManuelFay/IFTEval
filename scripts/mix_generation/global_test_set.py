import json
import os

PATH = "./outputs/fractial_mixes/"

test_set = []
test_dict = {}
for dir in sorted(os.listdir(PATH)):
    if dir.startswith("fractial_") and os.path.isdir(f"{PATH}/{dir}"):
        with open(f"{PATH}/{dir}/{dir}_only_test.jsonl") as f:
            data = [json.loads(line) for line in f][:20]
            assert len(data) == 20
            test_dict[dir] = data
            test_set.extend(data)

with open(f"{PATH}/fractial_test_set.jsonl", "w") as f:
    for line in test_set:
        f.write(json.dumps(line) + "\n")

with open(f"{PATH}/fractial_test_set.json", "w") as f:
    json.dump(test_dict, f, indent=4)


SLURM_TEMPLATE = """
sbatch --job-name={job_name} --nodes=1 --time=24:00:00 -p gpua100 --gres=gpu:1 --mem-per-cpu=32G --cpus-per-task=8 \
       --output={job_name}.out     --error={job_name}.err \
      --wrap="python model_training/offline_generate.py --max_new_tokens=256\
        --batch_size 8\
        --lora_weights models/{model_name}_{n} --data_path  data/fractial_test_set.jsonl --output_dir results_cross"
"""

for n in [0, 1000, 10, 100]:
    for model_name in test_dict.keys():
        with open(f"{PATH}/test_slurm_launch.sh", "a+") as f:
            f.write(SLURM_TEMPLATE.format(model_name=model_name, n=n, job_name=model_name[9:]))
