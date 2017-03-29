import argparse
import json
import numpy as np
import os
import pickle
import tensorflow as tf

import network

class Evaler:

    def __init__(self, save_path, is_verbose, batch_size=1):
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
            saver.restore(sess, os.path.join(save_path, "model"))

    def probs(self, inputs):
        model = self.model
        probs, = self.session.run([model.probs], model.feed_dict(inputs))
        return probs

    def predict(self, inputs):
        probs = self.probs(inputs)
        return np.argmax(probs, axis=1)

def predict_record(record_id, model_path):
    evaler = Evaler(model_path)

    ldr_path = os.path.join(model_path, "loader.pkl")
    with open(ldr_path, 'rb') as fid:
        ldr = pickle.load(fid)

    inputs = ldr.load_preprocess(record_id)
    outputs = evaler.predict([inputs])
    return ldr.int_to_class(outputs[0])

def main():
    parser = argparse.ArgumentParser(description="Evaluater Script")
    parser.add_argument("model_path")
    parser.add_argument("record")

    args = parser.parse_args()
    prediction = predict_record(args.record, args.model_path)
    print(prediction)

if __name__ == "__main__":
    main()

