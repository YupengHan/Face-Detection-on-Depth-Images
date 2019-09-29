#!/bin/bash
export OMP_NUM_THREADS=2
#export PYTHONPATH=/home/xingduan/YupengHan/nn/modules/:/home/xingduan/caffe_parallel/python/:$PYTHONPATH
export PYTHONPATH=/home/xingduan/YupengHan/caffe_modules/:/home/xingduan/caffe_parallel/python/:$PYTHONPATH

now=$(date +"%Y%m%d_%H%M%S")
if [ ! -d "./logs" ];then
    mkdir logs
fi

if [ ! -d "./models" ];then
    mkdir models
fi

/home/xingduan/caffe_parallel/.build_release/tools/caffe train -solver solver.prototxt -gpu 0 2>&1 | tee logs/Dep_48-$now.log
