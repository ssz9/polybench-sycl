CUR_DIR=${PWD}
cd ${HOME}/online/shenzt/swclcc
source ./set-swcl-18.sh
cd ${CUR_DIR}

MPIHOME=${HOME}/online/shenzt/repos/CGFDM3D-SYCL/deps/mpi_20230630_SEA
PROJHOME=${HOME}/online/shenzt/repos/CGFDM3D-SYCL/deps/proj-8.1.0_sw
SQLITEHOME=${HOME}/online/shenzt/repos/CGFDM3D-SYCL/deps/sqlite3_sw

export LD_LIBRARY_PATH=${PROJHOME}/lib64:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${SQLITEHOME}/lib:${LD_LIBRARY_PATH}
export LD_LIBRARY_PATH=${MPIHOME}/lib/single_dynamic:${MPIHOME}/lib/mpi_depend_libs:${LD_LIBRARY_PATH}

# export GIT_SSH='/home/export/online1/mdt00/shisuan/swpacman/shenzt/ssh/gitssh.sh'
export https_proxy=http://127.0.0.1:10808
export http_proxy=http://127.0.0.1:10808