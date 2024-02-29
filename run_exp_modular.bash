#!/bin/bash

# all auckland runs:
python3 synthetic_dataset.py --location "auckland" --nsamples 10 --seed 1
python3 synthetic_dataset.py --location "auckland" --nsamples 10 --seed 2
python3 synthetic_dataset.py --location "auckland" --nsamples 10 --seed 42
python3 synthetic_dataset.py --location "auckland" --nsamples 10 --seed 111

nohup python3 exp_modular.py --location "auckland" --module "all" --nsamples 10 --seed 1 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "auckland" --module "all" --nsamples 10 --seed 2 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "auckland" --module "all" --nsamples 10 --seed 42 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "auckland" --module "all" --nsamples 10 --seed 111 > /dev/null 2>&1 &

# all blackstone runs:
python3 synthetic_dataset.py --location "blackstone" --nsamples 10 --seed 1
python3 synthetic_dataset.py --location "blackstone" --nsamples 10 --seed 2
python3 synthetic_dataset.py --location "blackstone" --nsamples 10 --seed 42
python3 synthetic_dataset.py --location "blackstone" --nsamples 10 --seed 111

nohup python3 exp_modular.py --location "blackstone" --module "all" --nsamples 10 --seed 1 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "blackstone" --module "all" --nsamples 10 --seed 2 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "blackstone" --module "all" --nsamples 10 --seed 42 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "blackstone" --module "all" --nsamples 10 --seed 111 > /dev/null 2>&1 &

# all boston runs:
python3 synthetic_dataset.py --location "boston" --nsamples 10 --seed 1
python3 synthetic_dataset.py --location "boston" --nsamples 10 --seed 2
python3 synthetic_dataset.py --location "boston" --nsamples 10 --seed 42
python3 synthetic_dataset.py --location "boston" --nsamples 10 --seed 111

nohup python3 exp_modular.py --location "boston" --module "all" --nsamples 10 --seed 1 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "boston" --module "all" --nsamples 10 --seed 2 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "boston" --module "all" --nsamples 10 --seed 42 > /dev/null 2>&1 &
nohup python3 exp_modular.py --location "boston" --module "all" --nsamples 10 --seed 111 > /dev/null 2>&1 &
