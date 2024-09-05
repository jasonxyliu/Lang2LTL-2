# Spatio-temporal Language Grounding
Codebase for the paper Lang2LTL-2: Grounding Spatiotemporal Navigation Commands Using Large Language and Vision-Language Models [website](https://spatiotemporal-ground.github.io/).


# Installation
```
conda create -n ground python=3.9 dill matplotlib plotly scipy scikit-learn utm
conda activate ground
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
conda install -c conda-forge pyproj
conda install -c conda-forge spot
```


# Generate Synthetic Dataset for Evaluation
```
python synthetic_dataset.py --location <LOCATION>
```


# Lifted Command Translation Module
Please download finetuned T5-base model weights at [drive](https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing).


# Data
Please download data from [drive](https://drive.google.com/drive/folders/1gWomkuVqxLU01ftzF34bEacJBeUwBMOf?usp=sharing).


# Citation
```
@inproceedings{liu2024lang2ltl2,
  title     = {Lang2LTL-2: Grounding Spatiotemporal Navigation Commands Using Large Language and Vision-Language Models},
  author    = {Liu, Jason Xinyu and Shah, Ankit and Konidaris, George and Tellex, Stefanie and Paulius, David},
  booktitle = {IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  year      = {2024},
  url       = {https://arxiv.org/abs/},
}
```
