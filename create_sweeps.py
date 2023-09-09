import wandb
import os


lsf_file_template = """
#!/usr/bin/env bash
#BSUB -J gpu_job                              # CHANGE JOB NAME
#BSUB -q waic-long                            # QUEUE TO RUN IN
#BSUB -gpu num=1:j_exclusive=yes:gmodel=TeslaV100_SXM2_32GB              # NUM OF GPUS
#BSUB -R rusage[mem=25000]                     # MEMORY IN MB
#BSUB -R affinity[thread*4]                   # CPU THREADS


# WORKAROUND
if [ -f ~/.bash_profile ]; then
  . ~/.bash_profile
elif [ -f ~/.profile ]; then
  . ~/.profile
fi

# ACTIVATE YOUR CONDA ENVIRONMENT
module load CUDA/11.3.1 miniconda/4.10.3_environmentally
conda activate dr

# RUN YOUR CODE
wandb agent dataset_extraction/Dataset_Extraction/{0}
"""

manual_done = [(500, 10), (100, 500), (10, 50), (5, 10)] # (WIDTH, N)
sweep_project = 'Dataset_Extraction'

WIDTHS = [500, 100, 50, 10, 5]
TOTAL_SAMPLES = [10, 50, 100, 300, 500]

sweep_config = {
   'method': 'bayes',
    'metric': {
        'goal': 'minimize',
        'name': 'extraction score'
    }, 
    'early_terminate': {
        'eta': 2,
        'min_iter': 10000,
        'type': 'hyperband'
    },
    'parameters': {
        'data_per_class_train': {'value': 50},
        'extraction_data_amount_per_class': {'value': 100},
        'extraction_epochs': {'value': 50000},
        'extraction_evaluate_rate': {'value': 1000},
        'extraction_init_scale': {
            'distribution': 'log_uniform_values',
            'max': 0.1,
            'min': 1e-06
        },
        'extraction_lr': {
            'distribution': 'log_uniform_values',
            'max': 1,
            'min': 1e-05,
        },
        'extraction_min_lambda': {
            'distribution': 'uniform',
            'max': 0.5,
            'min': 0.01,
        },
        'extraction_model_relu_alpha': {
            'distribution': 'uniform',
            'max': 500,
            'min': 10,
        },
        'model_hidden_list': {'value': '[1000,1000]'},
        'model_init_list': {'value': '[0.001,0.001]'},
        'pretrained_model_path': {'value': 'PLACEHOLDER.pth'},
        'problem': {'value': 'cifar10_vehicles_animals'},
        'proj_name': {'value': 'PLACEHOLDER'},
        'run_mode': {'value': 'reconstruct'},
        'wandb_active': {'value': True}
  },
  'program': 'Main.py'
}

os.makedirs('bsub_configs', exist_ok=True)
for width in WIDTHS:
    for n in TOTAL_SAMPLES:
        if (width, n) in manual_done:
            continue
        samples_per_class = n // 2

        sweep_config["name"] = f'cifar10_vehicles_animals_d{samples_per_class}_width{width}_wd0'
        sweep_config["parameters"]["proj_name"]["value"] = sweep_config["name"]
        sweep_config["parameters"]["pretrained_model_path"]["value"] = f"weights-cifar10_vehicles_animals_d{samples_per_class}_width{width}_wd0.pth"
        sweep_config["parameters"]["model_hidden_list"]["value"] = f"[{width},{width}]"
        sweep_config["parameters"]["extraction_data_amount_per_class"]["value"] = samples_per_class * 2
        sweep_config["parameters"]["data_per_class_train"]["value"] = samples_per_class
        sweep_id = wandb.sweep(sweep_config, entity="dataset_extraction", project=sweep_project)
        print(f'Created sweep, ID={sweep_id}, width={width}, N={n}')
        with open(os.path.join('bsub_configs', f'run_gpu_width{width}_d{samples_per_class}.lsf'), 'w') as f:
            f.write(lsf_file_template.format(sweep_id))
