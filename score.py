import argparse
import json
import numpy as np
import os
import sklearn.metrics as skm

import loader
import test

def print_scores(labels, predictions, classes):
    report = skm.classification_report(
                labels, predictions,
                target_names=classes,
                digits=3)
    macro_scores = skm.precision_recall_fscore_support(
                        labels,
                        predictions,
                        average='macro')
    print(report)
    print("Macro Average F1: {:.3f}".format(macro_scores[2]))

def main():
    parser = argparse.ArgumentParser(description="Evaluater Script")
    parser.add_argument("model_path")

    args = parser.parse_args()

    batch_size = 8
    evaler = test.Evaler(args.model_path,
                 batch_size=batch_size)

    # TODO, (awni), would be good to simplify loading and
    # not rely on random seed for validation set.
    config_file = os.path.join(args.model_path, "config.json")
    with open(config_file, 'r') as fid:
        config = json.load(fid)
    data_conf = config['data']
    ldr = loader.Loader(data_conf['path'],
                        batch_size,
                        seed=data_conf['seed'])

    predictions = []
    labels = []
    for batch in ldr.val:
        preds = evaler.predict(batch[0])
        predictions.append(preds)
        labels.append(batch[1])
    predictions = np.hstack(predictions)
    labels = np.hstack(labels)

    print_scores(labels, predictions, ldr.classes)

if __name__ == "__main__":
    main()

