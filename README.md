# Madrona RL Environments

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bsarkar321/madrona_rl_envs/blob/master/overcooked_compiled_colab.ipynb)

Implementation of various RL Environments using the [Madrona game engine](https://madrona-engine.github.io/). You can also train an Overcooked agent in 2 minutes using the Google Colab link above. You can also read our [post](https://bsarkar321.github.io/blog/overcooked_madrona/index.html) about the process of porting Overcooked to Madrona.

## Environments

This repo contains implementations of various RL environments. Please read the respective README files to test the installation.

- [Balance Beam](src/balance_beam_env/README.org), a very simple 2-player cooperative game. This is a good starting point if you want to understand how to implement a simple environment in Madrona.
- [Cartpole](src/cartpole_env/README.org), the classic single-player RL environment.
- [Hanabi](src/hanabi_env/README.org), a standard environment for cooperative multi-agent RL with partial observability.
- [Overcooked](src/overcooked_env/README.org), a 2D grid-based environment for cooperative multi-agent RL based on the Overcooked game.
- [Overcooked (JS-compatible)](src/overcooked2_env/README.org), an older implementation of Overcooked that is compatible with a javascript interface. This is the version used in the Colab notebook.

## Requirements

To use Madrona with GPU, you need a CUDA version of at least 12.0 and a cmake version of at least 3.18. For these environments, you also need to have conda environments (miniconda/anaconda).

To install miniconda (from miniconda3 instructions):
```
mkdir miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm miniconda3/miniconda.sh
miniconda3/bin/conda init bash
# restart shell afterwards
```


## Installation

```
conda create -n madrona python=3.10
conda activate madrona
pip install torch numpy tensorboard

git clone https://github.com/bsarkar321/madrona_rl_envs
cd madrona_rl_envs
git submodule update --init --recursive
mkdir build
cd build
cmake ..
make -j
cd ..

pip install -e .

pip install -e overcooked_ai
pip install -e oldercooked_ai
```

NOTE: For cmake, you may need to specify the cuda tookit directory as follows:

```
cmake -D CUDAToolkit_ROOT=/usr/local/cuda-12.0 ..
```
