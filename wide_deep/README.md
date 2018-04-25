# config data info 
- support multi-value feature
model_info.txt file format
```
eg:  #<column name>	<column size> <hash size> <data type>
label	1	10000	int
ps	1	10000	string
uhy	100	10000	string
```

# Local Run
```
# train
sh local_run.sh

# Eval
sh local_run.sh eval

# Export
sh local_run.sh export
```

# Distribute Run
```
# train
sh distribute_run.sh

# Eval
sh distribute_run.sh eval

# Export
sh distribute_run.sh export
```

# Tensorboard
```
sh tensorboard.sh
```



