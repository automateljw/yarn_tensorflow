from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys, os
import collections
from config import FLAGS

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder
            
FIELD_OUTER_DELIM='\030'
FIELD_INNER_DELIM='\031'

cur_dir = os.path.dirname(os.path.realpath(__file__))
module_decode_file_serve = tf.load_op_library(os.path.join(cur_dir, 'decode_file_serve_op.so'))

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def read_model_info(filename):
    column_name = []
    column_size = []
    column_type = []
    hash_size = []
    with open(filename) as fd:
        line = fd.readline()
        skip_feature = line.strip().split(':')[1].split(',')
        for line in fd:
            _name, _size, _hash_size, _type = line.strip().split('\t')
            column_name.append(_name)
            column_size.append(int(_size))
            column_type.append(_type)
            hash_size.append(int(_hash_size))
    return column_name, column_size, column_type, hash_size, skip_feature

_COLUMN_NAMES, _COLUMN_SIZES, _COLUMN_TYPES, _HASH_SIZE, SKIP_FEATURE = read_model_info(FLAGS.model_info)

_COLUMN_DEFAULTS = map(lambda x: [0] if x == 'int' else [''], _COLUMN_TYPES)
_HASH_SIZE_DICT = dict(zip(_COLUMN_NAMES, _HASH_SIZE))

#SKIP_FEATURE = ['lastcate', 'net_wifi', 'cu', 'li', 'ip3']
print("[skip features] ", SKIP_FEATURE)
print(_COLUMN_NAMES)
print(_COLUMN_SIZES)
print(_COLUMN_TYPES)

def input_fn(data_dir, num_epochs, shuffle, batch_size, return_iterator=False):
  """Generate an input function for the Estimator."""
  data_files = tf.gfile.Glob(data_dir)
  assert len(data_files), ("%s has can't match any file!" % data_dir) 
  
  def parse_line(value):
    columns = module_decode_file_serve.decode_file_serve(value, 
            record_defaults=_COLUMN_DEFAULTS,
            output_size=_COLUMN_SIZES,
            field_outer_delim=FIELD_OUTER_DELIM,
            field_inner_delim=FIELD_INNER_DELIM)
    features = dict(zip(_COLUMN_NAMES, columns))
    labels = features.pop('label')
    return features, labels

  # Extract lines from input files using the Dataset API.
  dataset = tf.data.TextLineDataset(data_files)

  if shuffle: dataset = dataset.shuffle(buffer_size=batch_size*3)
  #dataset = dataset.map(parse_line, num_parallel_calls=1)

  # We call repeat after shuffling, rather than before, to prevent separate
  # epochs from blending together.
  if num_epochs: dataset = dataset.repeat(num_epochs)
  dataset = dataset.batch(batch_size)
  dataset = dataset.map(parse_line)
  #dataset = dataset.map(parse_line).prefetch(batch_size*2)

  if return_iterator == False:
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next("iterator")
    return features, labels
  else:
    iterator = dataset.make_initializable_iterator()
    features, labels = iterator.get_next("iterator")
    return iterator, features, labels

def build_model_columns():
  """Builds a set of wide and deep feature columns."""
  wide_fc = {}
  deep_fc = {}
  for feat in _COLUMN_NAMES[1:]:
    if feat in SKIP_FEATURE : continue
    print("feature:",feat)
    wide_fc[feat] = tf.feature_column.categorical_column_with_hash_bucket(feat, hash_bucket_size=_HASH_SIZE_DICT[feat])
    deep_fc[feat] = tf.feature_column.embedding_column(wide_fc[feat], dimension=24)

  # Wide columns and deep columns.
  wide_columns = [ v for k,v in wide_fc.items() ]
  deep_columns = [ v for k,v in deep_fc.items() ]

  return wide_columns, deep_columns

# export model input fn
def string_placeholder_serving_input_receiver_fn():
  column_info = collections.OrderedDict(zip(_COLUMN_NAMES[1:], _COLUMN_SIZES[1:]))
  #ph_list = [tf.placeholder(shape=[None, size], dtype=tf.string, name=name) for name, size in column_info.items() if name not in SKIP_FEATURE]
  #features = dict(zip(_COLUMN_NAMES[1:], ph_list))
  features = {}
  for name, size in column_info.items():
    if name in SKIP_FEATURE: continue
    print("feature: ", name)
    features[name] = tf.placeholder(shape=[None, size], dtype=tf.string, name=name)
  return tf.estimator.export.ServingInputReceiver(features, features)

def string_decode_serving_input_receiver_fn():
  string_placeholder = tf.placeholder(shape=[None, 1], dtype=tf.string)

  columns = module_decode_file_serve.decode_file_serve(string_placeholder,
          record_defaults=_COLUMN_DEFAULTS,
          output_size=_COLUMN_SIZES, 
          field_outer_delim=FIELD_OUTER_DELIM,
          field_inner_delim=FiELD_INNER_DELIM)
  features = dict(zip(_COLUMN_NAMES, columns))
  features.pop('label')

  return tf.estimator.export.ServingInputReceiver(features, string_placeholder)

def input_test():
  # test
  iterator, features, labels = input_fn('train_sample.txt', 1, False, 3, True)

  builder = _LazyBuilder(features)
  wide_columns, deep_columns = build_model_columns()

  print( [x.name for x in wide_columns])
  sparse_columns = [fc._get_sparse_tensors(builder).id_tensor for fc in wide_columns]
  X = tf.sparse_concat(1, sparse_columns)

  ps_index = [ x.name for x in wide_columns].index('ps')
  age_index = [ x.name for x in wide_columns].index('ag')
  rtuhy_index = [ x.name for x in wide_columns].index('rtuhy')
  hyfreq_index = [ x.name for x in wide_columns].index('hyfreq')
  ps_column_tensor = wide_columns[ps_index]._get_sparse_tensors(builder).id_tensor
  age_wide_column_tensor = wide_columns[age_index]._get_sparse_tensors(builder).id_tensor
  rtuhy_wide_column_tensor = wide_columns[rtuhy_index]._get_sparse_tensors(builder).id_tensor
  hyfreq_wide_column_tensor = wide_columns[hyfreq_index]._get_sparse_tensors(builder).id_tensor
  age_column_tensor = deep_columns[2]._get_dense_tensor(builder)
  #ps_embedding_tensor = deep_columns[ps_index]._get_dense_tensor(builder)

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    #print(sess.run([features, labels]))

    sess.run(iterator.initializer)
    print('[X]', sess.run(X))

    sess.run(iterator.initializer)
    print('age features eval', sess.run([features['ag']]))
    sess.run(iterator.initializer)
    print('rtuhy features eval', sess.run([features['rtuhy']]))
    sess.run(iterator.initializer)
    print('hyfreq features eval', sess.run([features['hyfreq']]))
    
    #sess.run(iterator.initializer)
    #print('[age category_column]', sess.run(age_wide_column_tensor))
    #sess.run(iterator.initializer)
    #print('[age embedding_column]', sess.run([age_column_tensor]))

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  input_test()
