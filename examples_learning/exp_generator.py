import os
import random

import create_learn_dataset as our_generator

""" example of complete argument list
-t --seed_true 123 
--blank_input_filepath "./coffee1_blank.pl" 
--true_filepath "./coffee1_123/coffee1_true.pl"
--true_wo_dec_filepath "./coffee1_123/coffee1_true_wo_dec.pl"
-d --seed_traj 1000
--dataset_folder "./coffee1_123/"
--dataset_size 100
--traj_length 10
-i --seed_init 2151
--init_filepath "./coffee1_123/coffee1_init2151.pl"
"""

exp_version = "coffee3"
filepath_blank = "./coffee3_blank.pl"
nb_trajectories = [10,100]
trajectory_length = [5]
nb_init_models = 10
seed_true = 187
seed_traj = 1337
seed_init = 42
exp_folder = f"./{exp_version}_{seed_true}"

if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)

# create true
args = [
    "-t", # true model
    "--seed_true", f"{seed_true}",
    "--blank_input_filepath", f"{filepath_blank}",
    "--true_filepath", f"{exp_folder}/{exp_version}_true.pl",
    "--true_wo_dec_filepath", f"{exp_folder}/{exp_version}_true_wo_dec.pl",
]
our_generator.main(args)

# create datasets
for dataset_size in nb_trajectories:
    for traj_len in trajectory_length:
        args = [
            "--true_wo_dec_filepath", f"{exp_folder}/{exp_version}_true_wo_dec.pl",
            "-d", # dataset
            "--seed_traj", f"{seed_traj}",
            "--dataset_folder", f"{exp_folder}/",
            "--dataset_size", f"{dataset_size}",
            "--traj_length", f"{traj_len}",
        ]
        our_generator.main(args)
        seed_traj += 1


# create init models
for idx in range(nb_init_models):
    args = [
        "-i",  # init
        "--seed_init", f"{seed_init}",
        "--init_filepath", f"{exp_folder}/{exp_version}_init{seed_init}.pl",
        "--blank_input_filepath", f"{filepath_blank}"
    ]
    our_generator.main(args)
    seed_init += 1


