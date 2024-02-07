# Spatio-temporal Language Grounding
Codebase for the paper Grounding Spatio-temporal Language Commands.

# Installation
```
conda create -n ground python=3.9 dill matplotlib plotly scipy scikit-learn utm
conda activate ground
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
conda install -c conda-forge pyproj
```

# Lifted Command Translation Module
Download finetuned T5-base model weights at [drive](https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing)
