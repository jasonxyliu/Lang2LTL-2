# Spatio-temporal Language Grounding
Codebase for the paper Grounding Spatio-temporal Language Commands.

# Installation
```
conda create -n ground python=3.9 dill matplotlib plotly scipy scikit-learn tqdm openai pandas
conda activate ground
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia  # GPU
conda install pytorch torchdata -c pytorch  # CPU
conda install -c conda-forge pyproj
pip install openai utm
```

# Data
Please download data from [drive](https://drive.google.com/drive/folders/1gWomkuVqxLU01ftzF34bEacJBeUwBMOf?usp=sharing).


# Generate Language Grounding Dataset for Evaluation
```
python dataset_full.py --location <LOCATION> --nsamples <N_PER_TEMPORAL_PATTERN> --seed <SEED>
```


# Lifted Command Translation Module
Download finetuned T5-base model weights at [drive](https://drive.google.com/drive/folders/1rZl8tblyVj-pZZW4OgbO1NJwMIT2fwx9?usp=sharing)
