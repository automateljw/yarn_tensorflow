called_file=${BASH_SOURCE[0]}
file_abs_path=`readlink -f $called_file`
_DIR=`dirname $file_abs_path`
###################################################
## 配置WK_DIR
WK_DIR=`dirname $_DIR`

source ${WK_DIR}/conf/default.conf
##################################

## 参数配置
num_ps=10
num_worker=50
model_type=wide_and_deep
batch_size=30000
## each worker will run train_epochs 
train_epochs=2
model_dir=$MODEL_PATH/model
export_dir=model_export
model_info=$WK_DIR/wide_deep/model_info.txt

echo "=================="
echo "model_dir = ${model_dir}"
echo "=================="

if [[ "$1" == "continue" ]]; then
    echo "[Continue Training]"
elif [[ $1 == "export" ]]; then
    source ${HADOOP_HOME}/libexec/hadoop-config.sh
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${JAVA_HOME}/jre/lib/amd64/server:${HADOOP_HOME}/lib/native
    export CLASSPATH=$(${HADOOP_HDFS_HOME}/bin/hadoop classpath --glob)
    echo "[Export SavedModel]"
    python src_cpu/wide_deep.py --model_dir=$model_dir --export_dir=${export_dir} --model_type=wide_and_deep  --work_mode=export
    exit 0
else
    hdfs dfs -rmr $model_dir
    hdfs dfs -mkdir -p $model_dir
    hdfs dfs -chmod 777 $model_dir
fi

TensorFlow_Submit  \
--appName ${MODEL_NAME} \
--archives=hdfs://ns3-backup/user/hero/tool/Python.zip#Python \
--files=${model_info},${WK_DIR}/wide_deep/src_cpu \
--ps_memory=10000 \
--worker_memory=10000 \
--worker_cores 4 \
--num_ps ${num_ps} \
--num_worker ${num_worker} \
--mode_local=false \
--tensorboard=false \
--data_dir=None \
--train_dir=${model_dir} \
--command=Python/bin/python src_cpu/wide_deep.py model_info=model_info.txt model_dir=${model_dir} train_data=${TRAIN_PATH}/0* test_data=${TEST_PATH}/*/0* batch_size=${batch_size} model_type=${model_type} train_epochs=${train_epochs}
