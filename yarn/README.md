DMLC YARN AppMaster
===================
* This folder contains Application code to allow rabit run on Yarn.
* See [tracker](../) for job submission.
  - run ```./build.sh``` to build the jar, before using the script

## Prepare
- yarn cluster
-- hdfs prefix
-- queue name

## build dmlc-yarn.jar
```
sh build.sh
```

## distributed shell on yarn
submit application to yarn
```
sh test_task.sh
hadoop jar  dmlc-yarn.jar org.apache.hadoop.yarn.dmlc.Client \
    -file dmlc-yarn.jar \
    -file wrap.sh \
	-defaultfs "hdfs://ns3-backup" \
    -tempdir "hdfs://ns3-backup/user/hero/tmp" \
    -queue "root.optimus.hero.hero" \
    sh wrap.sh
``` 

