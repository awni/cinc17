import argparse
import collections
import glob
import logging
import numpy as np
import os
import pickle
import random
import scipy.io as sio

logger = logging.getLogger("Loader")

class Loader:
    """
    Loader class for feeding data to the network. This class loads the training
    and validation data sets. Once the datasets are loaded, they can be batched
    and fed to the network. Example usage:

        ```
        data_path = <path_to_data>
        batch_size = 32
        ldr = Loader(data_path, batch_size)
        for batch in ldr.train:
            run_sgd_on(batch)
        ```

    This class is also responsible for normalizing the inputs.
    """

    def __init__(self, data_path, batch_size,
                 val_frac=0.2, seed=None,
                 augment=False, random_noise=False, random_samples=200):
        """
        :param data_path: path to the training and validation files
        :param batch_size: size of the minibatches to train on
        :param val_frac: fraction of the dataset to use for validation
                         (held out by record)
        :param seed: seed the rng for shuffling data
        :param augment: set to true to augment the training data
        """
        if not os.path.exists(data_path):
            msg = "Non-existent data path: {}".format(data_path)
            raise ValueError(msg)

        if seed is not None:
            random.seed(seed)

        self.batch_size = batch_size
        self.augment = augment

        self._train, self._val = load_all_data(data_path, val_frac, random_noise, random_samples)
        logger.debug("Training set has " + str(len(self._train)) + " samples")
        logger.debug("Validation set has " + str(len(self._val)) + " samples")

        self.compute_mean_std()
        self._train = [(self.normalize(ecg), l) for ecg, l in self._train]
        self._val = [(self.normalize(ecg), l) for ecg, l in self._val]

        label_counter = collections.Counter(l for _, l in self._train)

        classes = sorted([c for c, _ in label_counter.most_common()])
        self._int_to_class = dict(zip(range(len(classes)), classes))
        self._class_to_int = {c : i for i, c in self._int_to_class.items()}
        self.class_counts = [label_counter[c] for c in classes]

        self._train = self.batches(self._train)
        self._val = self.batches(self._val)

    def batches(self, data):
        """
        :param data: the raw dataset from e.g. `loader.train`
        :returns: Iterator to the minibatches. Each minibatch consists
                  of an (ecgs, labels) pair. The ecgs is a list of 1D
                  numpy arrays, the labels is a list of integer labels.
        """
        # Sort by length
        data = sorted(data, key = lambda x: x[0].shape[0])

        inputs, labels = zip(*data)
        labels = [self._class_to_int[l] for l in labels]
        batch_size = self.batch_size
        data_size = len(labels)

        end = data_size - batch_size + 1
        batches = [(inputs[i:i + batch_size], labels[i:i + batch_size])
                   for i in range(0, end, batch_size)]
        random.shuffle(batches)

        logger.debug("Data set {" + str(data_size) + " samples}, batch size {" \
                + str(batch_size) + "} -> " + str(len(batches)) + " batches")
        return batches

    def normalize(self, example):
        """
        Normalizes a given example by the training mean and std.
        :param: example: 1D numpy array
        :return: normalized example
        """
        return (example - self.mean) / self.std

    def compute_mean_std(self):
        """
        Estimates the mean and std over the training set.
        """
        n_samples = sum(len(w) for w, _ in self._train)
        mean_sum = np.sum([np.sum(w) for w, _ in self._train])
        mean = mean_sum / n_samples

        var_sum = np.sum([np.sum((w - mean)**2) for w, _ in self._train])
        var = var_sum / n_samples

        self.mean = mean.astype(np.float32)
        self.std = np.sqrt(var).astype(np.float32)

    @property
    def classes(self):
        return [self._int_to_class[i]
                for i in range(self.output_dim)]

    @property
    def output_dim(self):
        """ Returns number of output classes. """
        return len(self._int_to_class)

    @property
    def train(self):
        """ Returns the raw training set. """
        for ecgs, labels in self._train:
            if self.augment:
                ecgs = [transform(ecg) for ecg in ecgs]
            yield (ecgs, labels)

    @property
    def val(self):
        """ Returns the raw validation set. """
        return self._val

    def load_preprocess(self, record_id):
        ecg = load_ecg_mat(record_id + ".mat")
        return self.normalize(ecg)

    def int_to_class(self, label_int):
        """ Convert integer label to class label. """
        return self._int_to_class[label_int]

    def __getstate__(self):
        """
        For pickling.
        """
        return (self.mean,
                self.std,
                self._int_to_class,
                self._class_to_int,
                self.class_counts)

    def __setstate__(self, state):
        """
        For unpickling.
        """
        self.mean = state[0]
        self.std = state[1]
        self._int_to_class = state[2]
        self._class_to_int = state[3]
        self.class_counts = state[4]

def transform(ecg):
    # Amplitude invariance
    scale = random.uniform(0.2, 2.0)

    # Lead inversion
    flip = random.choice([-1.0, 1.0])

    # Shifting
    begin = random.randint(0, 1000)
    end = -random.randint(0, 1000)
    ecg = ecg[begin:end]

    return  ecg * flip * scale


def add_random_noise_samples(sample_count):
    retVal = []
    for i in range(0, sample_count):
        length_window = random.randint(3000, 18000)
        logger.debug("Random window length " + str(length_window))
        retVal.append((np.random.randint(low=-100, high=100, size=(length_window), dtype=np.int16), 'N'))
    logger.debug("First entry of random samples" + str(retVal[0]))

    return retVal



def load_all_data(data_path, val_frac, train_noise, random_samples):
    """
    Returns tuple of training and validation sets. Each set
    will contain a list of pairs of raw ecg and the
    corresponding label.
    """
    label_file = os.path.join(data_path, "REFERENCE-v2.csv")
    # Load record ids + labels
    with open(label_file, 'r') as fid:
        records = [l.strip().split(",") for l in fid]
    all_records = []

    # Load raw ecg
    for record, label in records:
        ecg_file = os.path.join(data_path, record + ".mat")
        ecg = load_ecg_mat(ecg_file)
        #logger.info( "ecg " + str(ecg.size))
        all_records.append((ecg, label))

    # Shuffle before train/val split
    random.shuffle(all_records)
    cut = int(len(all_records) * val_frac)
    train, val = all_records[cut:], all_records[:cut]
    logger.info(len(train))
    logger.info(train[0])
    if train_noise:
        logger.info("Adding random noise samples to training set")
        t = add_random_noise_samples(random_samples)
    return train, val

def load_ecg_mat(ecg_file):
    return sio.loadmat(ecg_file)['val'].squeeze()

def main():
    parser = argparse.ArgumentParser(description="Data Loader")
    parser.add_argument("-v", "--verbose",
            default = False, action = "store_true")
    parser.add_argument("-p", "--data_path",
            default="/deep/group/med/alivecor/training2017/")
    parser.add_argument("-b", "--batch_size", default=32)

    parsed_arguments = parser.parse_args()
    arguments = vars(parsed_arguments)

    is_verbose   = arguments['verbose']
    data_path    = arguments['data_path']
    batch_size   = arguments['batch_size']


    if is_verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    random.seed(2016)
    ldr = Loader(data_path, batch_size, random_noise=True, random_samples=200)
    logger.info("Length of training set {}".format(len(list(ldr.train))))
    logger.info("Length of validation set {}".format(len(ldr.val)))
    logger.info("Output dimension {}".format(ldr.output_dim))

    # Run a few sanity checks.
    count = 0
    for ecgs, labels in ldr.train:
        count += 1
        assert len(ecgs) == len(labels) == batch_size, "Invalid example count."
        assert len(ecgs[0].shape) == 1, "ECG array should be 1D"

if __name__ == '__main__':
    main()

