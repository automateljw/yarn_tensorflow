#!/bin/sh
called_file=${BASH_SOURCE[0]}
file_abs_path=`readlink -f $called_file`
_DIR=`dirname $file_abs_path`
###################################################
WK_DIR=`dirname $_DIR`

source ${WK_DIR}/conf/default.conf

model_dir=${MODEL_PATH}/model

source ${HADOOP_HOME}/libexec/hadoop-config.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native
export CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)
tensorboard --logdir $model_dir --port=33333
