# grounding
Codebase for the Spatio-temporal grounding paper

# Installation
```
conda create -n ground python=3.9 dill matplotlib plotly scipy scikit-learn utm
conda activate ground
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
conda install -c conda-forge pyproj
```
