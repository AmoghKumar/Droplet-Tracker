# Nanodroplet size tracking in BF microscopy

<p align="center">
  <a href="#about">About</a> •
  <a href="#examples">Examples</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#useful-links">Useful Links</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

This repository is for testing models for tracking nanodroplets in brightfield microscopy.


## Examples



## Installation

Create/Install the environment like this:

conda create -n sam2-env python=3.10 -y
conda activate sam2-env

Install PyTorch first:

CPU version
pip install torch torchvision
OR CUDA version (recommended if you have NVIDIA GPU)

Example for CUDA 12.1:

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

Then:

pip install -r requirements.txt

You will also need the SAM2 checkpoints/configs separately:

Download checkpoints from Meta’s SAM2 repo
Place them in:
checkpoints/

For example:

checkpoints/sam2.1_hiera_small.pt

Also clone/download the SAM2 config folder structure:

configs/sam2.1/sam2.1_hiera_s.yaml

## How To Use

To train a model, run the following command:

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Where `CONFIG_NAME` is a config from `src/configs` and `HYDRA_CONFIG_ARGUMENTS` are optional arguments.

## Useful Links:

You may find the following links useful:

- [Python Dev Tips](https://github.com/ebezzam/python-dev-tips): information about [pre-commits](https://pre-commit.com/), [Hydra](https://hydra.cc/docs/intro/), and other stuff for better python code development.

- [Awesome README](https://github.com/matiassingers/awesome-readme): a list of awesome README files for inspiration. Check the basics [here](https://github.com/PurpleBooth/a-good-readme-template).

- [Report branch](https://github.com/Blinorot/pytorch_project_template/tree/report): Guidelines for writing a scientific report/paper (with an emphasis on DL projects).

## Credits

This repository is based on a heavily modified fork of [pytorch-template](https://github.com/victoresque/pytorch-template) and [asr_project_template](https://github.com/WrathOfGrapes/asr_project_template) repositories.

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)
