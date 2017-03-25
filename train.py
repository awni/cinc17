from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import numpy as np
import os
import pickle
import random
import tensorflow as tf
import time

import loader
import network
import utils

logger = logging.getLogger("Train")

def run_epoch(model, data_loader, session, summarizer):
    summary_op = tf.summary.merge_all()

    for batch in data_loader.train:
        ops = [model.train_op, model.avg_loss,
               model.avg_acc, model.it, summary_op]
        res = session.run(ops, feed_dict=model.feed_dict(*batch))
        _, loss, acc, it, summary = res
        summarizer.add_summary(summary, global_step=it)
        if it == 50:
            model.set_momentum(session)
        if it % 100 == 0:
            msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
            print(msg.format(it, loss, acc))

def run_validation(model, data_loader, session, summarizer):
    it = model.it.eval(session)
    results = []
    for batch in data_loader.val:
        ops = [model.acc, model.loss]
        feed_dict = model.feed_dict(*batch)
        res = session.run(ops, feed_dict=feed_dict)
        results.append(res)
    acc, loss = np.mean(results, axis=0)
    summary = utils.make_summary("Dev Accuracy", float(acc))
    summarizer.add_summary(summary, global_step=it)
    summary = utils.make_summary("Dev Loss", float(loss))
    summarizer.add_summary(summary, global_step=it)
    msg = "Validation: Loss {:.3f}, Acc {:.3f}"
    print(msg.format(loss, acc))

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train driver")
    parser.add_argument("-v", "--verbose",
            default=False, action="store_true")
    parser.add_argument("-c", "--config_file",
            default="configs/test.json")

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose   = arguments['verbose']
    config_file  = arguments['config_file']

    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    
    with open(config_file) as fid:
        config = json.load(fid)

    random.seed(config['seed'])
    epochs = config['optimizer']['epochs']
    data_loader = loader.Loader(config['data']['path'],
                                config['model']['batch_size'],
                                seed=config['data']['seed'])

    model = network.Network(is_verbose)

    output_save_path = config['io']['output_save_path']
    if not os.path.exists(output_save_path):
        os.mkdir(output_save_path)

    config['model']['output_dim'] = data_loader.output_dim
    with open(os.path.join(output_save_path, "config.json"), 'w') as fid:
        json.dump(config, fid)
    with open(os.path.join(output_save_path, "loader.pkl"), 'wb') as fid:
        pickle.dump(data_loader, fid)

    with tf.Graph().as_default(), tf.Session() as sess:
        tf.set_random_seed(config['seed'])
        model.init_inference(config['model'])
        model.init_loss()
        model.init_train(config['optimizer'])
        tf.global_variables_initializer().run()
        saver = tf.train.Saver(tf.global_variables())
        summarizer = tf.summary.FileWriter(output_save_path, sess.graph)
        for e in range(epochs):
            start = time.time()
            run_epoch(model, data_loader, sess, summarizer)
            saver.save(sess, os.path.join(output_save_path, "model"))
            print("Epoch {} time {:.1f} (s)".format(e, time.time() - start))
            run_validation(model, data_loader, sess, summarizer)

if __name__ == '__main__':
    main()
