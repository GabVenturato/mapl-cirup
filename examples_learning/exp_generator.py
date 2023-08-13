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

exp_version = "coffee1"
filepath_blank = "./coffee1_blank.pl"
nb_trajectories = 100
trajectory_length = 10
nb_init_models = 10
seed_true = 123
seed_traj = 1000

# create true and dataset
args = [
    "-t", # true model
    "--seed_true", f"{seed_true}",
    "--blank_input_filepath", "./coffee1_blank.pl",
    "--true_filepath", f"./coffee1_{seed_true}/coffee1_true.pl",
    "--true_wo_dec_filepath", f"./coffee1_{seed_true}/coffee1_true_wo_dec.pl",
    "-d", # dataset
    "--seed_traj", f"{seed_traj}",
    "--dataset_folder", f"./coffee1_{seed_true}/",
    "--dataset_size", f"{nb_trajectories}",
    "--traj_length", f"{trajectory_length}",
]
our_generator.main(args)


# create init models
for idx in range(nb_init_models):
    init_seed = random.randint(0,200000000000)
    args = [
        "-i",  # init
        "--seed_init", f"{init_seed}",
        "--init_filepath", f"./coffee1_{seed_true}/coffee1_init{init_seed}.pl",
        "--blank_input_filepath", "./coffee1_blank.pl",
    ]
    our_generator.main(args)


