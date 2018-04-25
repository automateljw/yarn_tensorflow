from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import timeit
import numpy as np
import six

import tensorflow as tf
from tensorflow.python.training.session_run_hook import SessionRunHook
from tensorflow.python.training.session_run_hook import SessionRunArgs
#from tensorflow.python.training.session_run_hook import _as_graph_element

from tensorflow.core.framework.summary_pb2 import Summary
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util.event_pb2 import SessionLog
from tensorflow.python.client import timeline
from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.training import session_run_hook
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs
from tensorflow.python.training.summary_io import SummaryWriterCache

def _as_graph_element(obj):
  """Retrieves Graph element."""
  graph = ops.get_default_graph()
  if not isinstance(obj, six.string_types):
    if not hasattr(obj, "graph") or obj.graph != graph:
      raise ValueError("Passed %s should have graph attribute that is equal "
                       "to current graph %s." % (obj, graph))
    return obj
  if ":" in obj:
    element = graph.as_graph_element(obj)
  else:
    element = graph.as_graph_element(obj + ":0")
    #element = graph.as_graph_element(obj)
    # Check that there is no :1 (e.g. it's single output).
    try:
      graph.as_graph_element(obj + ":1")
    except (KeyError, ValueError):
      pass
    else:
      raise ValueError("Name %s is ambiguous, "
                       "as this `Operation` has multiple outputs "
                       "(at least 2)." % obj)
  return element


class PredictOutputHook(SessionRunHook):
    ''' save predict output to file 
    Args:
        output_file: string, filename
        tensors: dict, key should include "label" and "pred",
             value is tensor's name
    '''
    def __init__(self, output_file, tensors):
        """ Initializes a `PredictOutputHook`.

        Args:
            output_file: output file name.
        """
        self._output_file = output_file
        if not isinstance(tensors, dict):
            self._tag_order = tensors
            tensors = {item: item for item in tensors}
        else:
            self._tag_order = tensors.keys()
        self._tensors = tensors
        self._count = 0
        self._begin = timeit.default_timer()

    def begin(self):
        self.fd = tf.gfile.Open(self._output_file, 'w')
        self._current_tensors = {tag: _as_graph_element(tensor)
                                for (tag, tensor) in self._tensors.items()}
        #print(self._current_tensors)

    def before_run(self, run_context):
        #print(self._current_tensors)
        return SessionRunArgs(self._current_tensors)

    def after_run(self, run_context, run_values):
        #for tag in self._tag_order:
        #    print(tag)
        true = run_values.results['label']
        #pred = run_values.results['pred'][:,1]
        pred = run_values.results['pred']
        #print('ture',true, 'pred',pred)
        pred = np.reshape(pred, (len(pred),-1))
        ret = np.concatenate((true, pred), axis=1)
        self._count += len(ret)
        np.savetxt(self.fd, ret, fmt="%.1f %.6f")
        if self._count == 10 or self._count % 100 == 0:
            self._end = timeit.default_timer()
            print("predict examples: %d, time: %f sec" % (self._count, self._end -
                self._begin))

    def end(self, session):
        print("======== total examples: %d ============" % self._count)
        self.fd.close()



