# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example code for TensorFlow Wide & Deep Tutorial using tf.estimator API."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import shutil
import sys,os
import json
import numpy as np

import tensorflow as tf
from tensorflow.python.feature_column.feature_column import _LazyBuilder

from config import *
#import input
from input import input_fn, build_model_columns
from input import string_decode_serving_input_receiver_fn
from input import string_placeholder_serving_input_receiver_fn
from custom_hook import PredictOutputHook
from dnn_linear_combined import DNNLinearCombinedClassifier


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def build_estimator(model_dir, model_type):
  """Build an estimator appropriate for the given model type."""
  wide_columns, deep_columns = build_model_columns()
  #hidden_units = [100, 75, 50, 25]
  hidden_units = [256, 128, 64, 16]

  # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
  # trains faster than GPU for this model.
  run_config = tf.estimator.RunConfig().replace(
      session_config=tf.ConfigProto(device_count={'GPU': 0}),
      save_checkpoints_steps=5000)

  linear_optimizer = tf.train.FtrlOptimizer(learning_rate=0.004)
  dnn_optimizer = tf.train.AdagradOptimizer(learning_rate=0.003)

  
  if model_type == 'wide':
    return tf.estimator.LinearClassifier(
        model_dir=model_dir,
        feature_columns=wide_columns,
        config=run_config)
  elif model_type == 'deep':
    return tf.estimator.DNNClassifier(
        model_dir=model_dir,
        feature_columns=deep_columns,
        hidden_units=hidden_units,
        config=run_config)
  else:
    #return DNNLinearCombinedClassifier(
    #    model_dir=model_dir,
    #    linear_feature_columns=wide_columns,
    #    dnn_feature_columns=deep_columns,
    #    dnn_hidden_units=hidden_units,
    #    config=run_config)
    return tf.estimator.DNNLinearCombinedClassifier(
        model_dir=model_dir,
        linear_feature_columns=wide_columns,
        dnn_feature_columns=deep_columns,
        dnn_hidden_units=hidden_units,
        linear_optimizer=linear_optimizer,
        dnn_optimizer=dnn_optimizer,
        config=run_config)


def build_config():
  ''' for distribution run environment config '''
  ps_hosts = FLAGS.ps_hosts.split(",")
  worker_hosts = FLAGS.worker_hosts.split(",")
  chief_host = [worker_hosts[0]]
  worker_hosts = worker_hosts[2:]
  #print(type(chief_host), type(worker_hosts), type(ps_hosts))
  eprint('chief=',chief_host, 'worker=',worker_hosts, 'ps=',ps_hosts)

  task_type = FLAGS.job_name
  task_index = FLAGS.task_index
  if(task_type == 'worker'):
    if(task_index == 0):
      task_type = 'chief'
    elif(task_index == 1):
      task_type = 'evaluator'
      task_index = 0
    else:
      task_index -= 2
  
  cluster = {'chief': chief_host,
             'worker': worker_hosts,
             'ps': ps_hosts}
  os.environ['TF_CONFIG'] = json.dumps(
      {'cluster': cluster,
       'task': {'type': task_type, 'index': task_index}})

def print_param():
  eprint("=========== params ========")
  eprint("batch_size = %d" % FLAGS.batch_size)
  eprint("max_steps = %d" % FLAGS.max_steps)
  eprint("train_epochs = %d" % FLAGS.train_epochs)
  eprint("===========================")

def main(argv):
  # Clean up the model directory if present
  #shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
  
  # build TF_CONFIG env
  if FLAGS.job_name != 'None':
    eprint('run: distribute')
    build_config()
  else:
    eprint('run: local')

  print_param()

  model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
  logging_hook = tf.train.LoggingTensorHook({"logits": "head/predictions/logistic"}, every_n_iter=1)
  test_profiler_hook = tf.train.ProfilerHook(save_steps=10, output_dir='.')
  
  if FLAGS.work_mode == 'train_and_eval':
    #train_hooks = [train_profiler_hook]
    train_hooks = None
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: input_fn(FLAGS.train_data,
        num_epochs=FLAGS.train_epochs, shuffle=True, batch_size=FLAGS.batch_size), max_steps=FLAGS.max_steps,
        hooks=train_hooks)
    #eval_hooks = [logging_hook]
    eval_hooks = None
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: input_fn(FLAGS.test_data, num_epochs=None,
        shuffle=True, batch_size=FLAGS.batch_size), steps=None, hooks=eval_hooks, start_delay_secs=120,
        throttle_secs=300)
        #FLAGS.batch_size), steps=50)
        #FLAGS.batch_size), steps=50, hooks=[logging_hook])
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
  elif FLAGS.work_mode == 'eval':
    if FLAGS.model_type == "wide":
        label_pred = {"label": "linear/head/labels", "pred": "linear/head/predictions/probabilities"}
    elif FLAGS.model_type == "wide_and_deep":
        #label_pred = {"label": "head/labels", "pred": "head/predictions/probabilities"}
        label_pred = {"label": "head/labels", "pred": "head/predictions/logits"}
    output_hook = PredictOutputHook('output_pred.txt', label_pred)
    eval_hooks = [output_hook, test_profiler_hook]
    results = model.evaluate(input_fn=lambda: input_fn(
        FLAGS.test_data, num_epochs=1, shuffle=False, batch_size=FLAGS.batch_size), steps=None, hooks=eval_hooks)
    for key in sorted(results):
      print('%s: %s' % (key, results[key]))
  elif FLAGS.work_mode == 'export':
    export_dir = model.export_savedmodel(FLAGS.export_dir, string_placeholder_serving_input_receiver_fn, as_text=True)
    eprint(export_dir)
    np.savetxt(os.path.join(FLAGS.export_dir, 'checkpoint'), [export_dir], fmt='%s')

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  #tf.logging.set_verbosity(tf.logging.DEBUG)

  #FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main)
