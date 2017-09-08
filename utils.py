
import tensorflow as tf
import numpy as np
import sklearn.metrics as skm

def cinc_score(labels, predictions):
    """
    Computes macro average of F1 but only for A, N and O.
    Does not use the ~ (noise) F1 in the macro average.
    See:
    https://groups.google.com/forum/#!topic/physionet-challenges/64O7nhp430Q
    """
    scores = skm.precision_recall_fscore_support(
                        labels,
                        predictions,
                        average=None)
    return np.mean(scores[2][:3])

def make_summary(tag, value):
    value = tf.Summary.Value(tag=tag, simple_value=value)
    return tf.Summary(value=[value])
