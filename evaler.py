import argparse
import json
import numpy as np
import os
import pickle
import tensorflow as tf

import network

class Evaler:

    def __init__(self, save_path, is_verbose=False,
                 batch_size=1, class_counts=None,
                 smooth=350): # TODO, awni, setup way to x-val smoothing param
        config_file = os.path.join(save_path, "config.json")

        with open(config_file, 'r') as fid:
            config = json.load(fid)
        config['model']['batch_size'] = batch_size

        self.model = network.Network(is_verbose)
        self.graph = tf.Graph()
        self.session = sess = tf.Session(graph=self.graph)
        with self.graph.as_default():
            self.model.init_inference(config['model'])
            tf.global_variables_initializer().run(session=sess)
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, os.path.join(save_path, "best_model.epoch"))

        if class_counts is not None:
            counts = np.array(class_counts)[None, :]
            total = np.sum(counts) + counts.shape[1]
            self.prior = (counts + smooth) / total
        else:
            self.prior = None

    def probs(self, inputs):
        model = self.model
        probs, = self.session.run([model.probs], model.feed_dict(inputs))
        if self.prior is not None:
            probs /= self.prior
        return probs

    def predict(self, inputs):
        probs = self.probs(inputs)
        return np.argmax(probs, axis=1)

def predict_record(record_id, model_path, prior=False):
    ldr_path = os.path.join(model_path, "loader.pkl")
    with open(ldr_path, 'rb') as fid:
        ldr = pickle.load(fid)

    if prior:
        evaler = Evaler(model_path, class_counts=ldr.class_counts)
    else:
        evaler = Evaler(model_path)

    inputs = ldr.load_preprocess(record_id)
    outputs = evaler.predict([inputs])
    return ldr.int_to_class(outputs[0])

def main():
    parser = argparse.ArgumentParser(description="Evaluater Script")
    parser.add_argument("-m", "--model_path")
    parser.add_argument("-r", "--record")
    parser.add_argument("-p", "--prior", action="store_true",
                        help="Decode with prior")

    args = parser.parse_args()
    prediction = predict_record(args.record, args.model_path,
                                prior=args.prior)
    print(prediction)

if __name__ == "__main__":
    main()

