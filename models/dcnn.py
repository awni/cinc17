
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl

import model

class DCNN(model.Model):

    def init_inference(self, config):
        self.output_dim = num_labels = config['output_dim']
        self.batch_size = batch_size = config['batch_size']

        self.inputs = inputs = tf.placeholder(tf.float32, shape=(batch_size, None))
        acts = tf.reshape(inputs, (batch_size, -1, 1, 1))
        self.keep_prob = tf.placeholder(tf.float32)
        self.drop_prob = config['dropout']

        for layer in config['conv_layers']:
            num_filters = layer['num_filters']
            kernel_size = layer['kernel_size']
            acts = tfl.convolution2d(acts, num_outputs=num_filters,
                                     kernel_size=[kernel_size, 1])
            acts = tf.nn.dropout(acts, self.keep_prob)

        # Activations should emerge from the convolution with shape
        # [batch_size, time (subsampled), 1, num_channels]
        acts = tf.squeeze(acts, squeeze_dims=[2])

        self.logits = tfl.fully_connected(acts, self.output_dim)
        self.probs = tf.nn.softmax(self.logits)

    def feed_dict(self, inputs, labels=None, test=False):
        """
        Generates a feed dictionary for the model's place-holders.
        *NB* inputs and labels are assumed to all be of the same
        lenght.
        Params:
            inputs : List of 1D arrays of wave segments
            labels (optional) : List of lists of integer labels
        Returns:
            feed_dict (use with feed_dict kwarg in session.run)
        """
        feed_dict = {self.inputs : _zero_pad_mat(inputs, np.float32)}
        if labels is not None:
            feed_dict[self.labels] = _zero_pad_mat(labels, np.int32)
            mask = [np.ones(len(label), dtype=np.float32)
                       for label in labels]
            feed_dict[self.mask] = _zero_pad_mat(mask, np.float32)
        if test:
            feed_dict[self.keep_prob] = 1.0
        else:
            feed_dict[self.keep_prob] = 1.0 - self.drop_prob

        return feed_dict

def _zero_pad_mat(vectors, dtype):
    max_len = max(len(vec) for vec in vectors)
    mat = np.zeros((len(vectors), max_len), dtype=dtype)
    for e, vec in enumerate(vectors):
        mat[e, :len(vec)] = vec
    return mat


