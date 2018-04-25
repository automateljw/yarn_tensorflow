#!/bin/sh
called_file=${BASH_SOURCE[0]}
file_abs_path=`readlink -f $called_file`
_DIR=`dirname $file_abs_path`
###################################################
## Config workspace dir
WK_DIR=`dirname $_DIR`

source ${WK_DIR}/conf/default.conf

model_info=model_info.txt
model_dir=model_local

# Config environment for tensorflow read hdfs file directly
source ${HADOOP_HOME}/libexec/hadoop-config.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native
export CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)


if [[ $1 == "eval" ]]; then 
    #python src_cpu/wide_deep.py --model_info=$model_info --model_dir=$model_dir --model_type=wide_and_deep --test_data=${TEST_PATH}/*/0* --work_mode=eval --batch_size=10000
    python src_cpu/wide_deep.py --model_info=$model_info --model_dir=$model_dir --model_type=wide_and_deep --test_data=news_head --work_mode=eval --batch_size=10000
elif [[ $1 == "export" ]]; then
    python src_cpu/wide_deep.py --model_info=$model_info --model_dir=model_local --model_type=wide_and_deep  --work_mode=export
else
    python src_cpu/wide_deep.py --model_info=$model_info --train_data=fydata/* --test_data=fydata/* --model_type=wide_and_deep --model_dir=model_local --train_epochs=1
    #python src_cpu/wide_deep.py --model_info=$model_info --train_data=${TRAIN_PATH}/0* --test_data=${TEST_PATH}/psid=*/0* --model_type=wide_and_deep --model_dir=model_local
fi
