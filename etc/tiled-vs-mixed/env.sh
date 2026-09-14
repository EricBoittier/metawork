source $HOME/venvs/petlr/bin/activate
export DATASETS=/capstor/store/cscs/swissai/aa002/rumiants/datasets
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
export GLIBC_TUNABLES=glibc.malloc.mmap_threshold=134217728
export CUDA_VISIBLE_DEVICES=0
cd $HOME/petlr/work/tiled_vs_mixed
