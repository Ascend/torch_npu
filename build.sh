#!/bin/bash

set -ex

git submodule sync
git submodule update --init --recursive

python setup.py clean

export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/devuser/workspace/pytorch/build/lib"

export DEBUG=1
#export BUILD_TEST=1
#export INSTALL_TEST=1
export _GLIBCXX_USE_CXX11_ABI=0
export USE_CUDA=0
export USE_XNNPACK=0
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}
python setup.py develop
