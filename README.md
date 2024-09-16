# Jupyter Notebooks for the 2024 CERN School of Computing Course on Machine Learning

https://indico.cern.ch/event/1376644/

## Installation Instructions

Some dependencies are rather hard to install for this workflow. The `requirements.txt` was tested with python `3.12`. 

1. create a `venv` by `python -m venv py311 --upgrade-deps`
2. setup that `venv` by `source py311/bin/activate`
3. (optional) install `uv` for faster installations
4. either do `uv pip install -r ./requirements.txt` or plain `python -m pip install -r ./requirements.txt`

If you like to train cpu-only, you have to install torch without CUDA support. This is best beformed between step 2 and 4 in the recipe above by running:
```shell
uv pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```
or without `uv`:
```shell
python -m pip install --extra-index-url https://download.pytorch.org/whl/cpu torch
```

## Usage on SWAN

TODO
