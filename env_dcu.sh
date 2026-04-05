module purge
module load compiler/rocm/dtk/24.04.3


source /public/home/pacman/spack/share/spack/setup-env.sh
. ${HOME}/envs/env_intel-llvm.sh

export PATH=/public/home/pacman/opt/ninja-1.12.1:$PATH

export https_proxy=http://localhost:10808
export http_proxy=http://localhost:10808

ulimit -c 0

