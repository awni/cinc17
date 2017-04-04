import argparse
import json
import numpy as np
import os
import sklearn.metrics as skm
import logging

import loader
import evaler

logger = logging.getLogger("Score")

def print_scores(labels, predictions, classes):
    accuracy = skm.accuracy_score(labels, predictions)
    report = skm.classification_report(
                labels, predictions,
                target_names=classes,
                digits=3)
    macro_scores = skm.precision_recall_fscore_support(
                        labels,
                        predictions,
                        average='macro')
    logger.info("Accuracy {:.3f}".format(accuracy))
    logger.info("\n"+report)
    logger.info("Macro Average F1: {:.3f}".format(macro_scores[2]))

def load_model(model_path, is_verbose, batch_size):

    # TODO, (awni), would be good to simplify loading and
    # not rely on random seed for validation set.
    config_file = os.path.join(model_path, "config.json")
    with open(config_file, 'r') as fid:
        config = json.load(fid)
    data_conf = config['data']
    ldr = loader.Loader(data_conf['path'],
                        batch_size,
                        seed=data_conf['seed'])

    evl = evaler.Evaler(model_path, is_verbose,
                 batch_size=batch_size,
                 class_counts=None)

    return evl, ldr

def eval_all(ldr, evl):
    predictions = []
    labels = []
    for batch in ldr.val:
        preds = evl.predict(batch[0])
        predictions.append(preds)
        labels.append(batch[1])
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)
    return predictions, labels

def main():
    parser = argparse.ArgumentParser(description="Evaluater Script")
    parser.add_argument("-m", "--model_path", default="~/")
    parser.add_argument("-v", "--verbose", default=False, action="store_true")

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose = arguments['verbose']
    model_path = arguments['model_path']
    batch_size = 8

    evl, ldr = load_model(model_path, is_verbose, batch_size)

    predictions, labels = eval_all(ldr, evl)
    print_scores(labels, predictions, ldr.classes)

if __name__ == "__main__":
    main()

