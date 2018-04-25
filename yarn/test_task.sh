hadoop jar  dmlc-yarn.jar org.apache.hadoop.yarn.dmlc.Client \
    -file dmlc-yarn.jar \
    -file wrap.sh \
    -tempdir "hdfs://ns3-backup/user/hero/tmp" \
    -defaultfs "hdfs://ns3-backup" \
    -queue "root.optimus.hero.hero" \
    sh wrap.sh
