from __future__ import absolute_import
from __future__ import division

import argparse
import json
import logging
import numpy as np
import os
import pickle
import random
import sklearn.metrics as skm
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
            logger.debug("Setting initial momentum in iteration " + str(it))

        msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
        logger.debug(msg.format(it, loss, acc))
        if it % 100 == 0:
            msg = "Iter {}: AvgLoss {:.3f}, AvgAcc {:.3f}"
            logger.info(msg.format(it, loss, acc))
    return acc

def run_validation(model, data_loader, session, summarizer):
    it = model.it.eval(session)
    predictions = []
    labels = []
    losses = []
    for batch in data_loader.val:
        ops = [model.probs, model.loss]
        feed_dict = model.feed_dict(*batch)
        probs, loss = session.run(ops, feed_dict=feed_dict)
        predictions.extend(np.argmax(probs, axis=1).tolist())
        labels.extend(batch[1])
        losses.append(loss)
    loss = np.mean(losses)
    acc = skm.accuracy_score(labels, predictions)
    mac_f1 = skm.precision_recall_fscore_support(
                        labels,
                        predictions,
                        average='macro')[2]
    summary = utils.make_summary("Dev Accuracy", float(acc))
    summarizer.add_summary(summary, global_step=it)
    summary = utils.make_summary("Dev Loss", float(loss))
    summarizer.add_summary(summary, global_step=it)
    msg = "Validation: Loss {:.3f}, Acc {:.3f}, Macro F1 {:.3f}"
    logger.info(msg.format(loss, acc, mac_f1))
    return acc

def main(argv=None):
    parser = argparse.ArgumentParser(description="Train driver")
    parser.add_argument("-v", "--verbose",
            default=False, action="store_true")
    parser.add_argument("-c", "--config_file",
            default="configs/train.json")

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
                    seed=config['data']['seed'],
                    augment=config['data'].get('augment', False))

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

        best_eval_acc = 0.0;

        for e in range(epochs):
            start = time.time()
            train_acc = run_epoch(model, data_loader, sess, summarizer)
            saver.save(sess, os.path.join(output_save_path, "model"))
            logger.info("Epoch {} time {:.1f} (s)".format(e, time.time() - start))
            eval_acc = run_validation(model, data_loader, sess, summarizer)

            if eval_acc > best_eval_acc:
                saver.save(sess, os.path.join(output_save_path, "best_model.epoch"))
                best_eval_acc = eval_acc
                logger.info("Best accuracy so far: " + str(best_eval_acc))

if __name__ == '__main__':
    main()
