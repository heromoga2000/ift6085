try:
    import cPickle
except ImportError:
    import pickle as cPickle
import gzip
import os

import numpy as np

def load_mnist():
    # Check if dataset is in the data directory.
    data_path = os.path.join(os.path.split(__file__)[0], "data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    dataset = 'mnist.pkl.gz'
    data_file = os.path.join(data_path, dataset)
    if os.path.isfile(data_file):
        dataset = data_file

    if (not os.path.isfile(data_file)) and dataset == 'mnist.pkl.gz':
        try:
            import urllib
            urllib.urlretrieve('http://www.google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        print('Downloading data from %s' % url)
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = cPickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_set_x, test_set_y = test_set
    test_set_x = test_set_x.astype('float32')
    test_set_y = test_set_y.astype('int32')
    valid_set_x, valid_set_y = valid_set
    valid_set_x = valid_set_x.astype('float32')
    valid_set_y = valid_set_y.astype('int32')
    train_set_x, train_set_y = train_set
    train_set_x = train_set_x.astype('float32')
    train_set_y = train_set_y.astype('int32')

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def load_orderly_mnist_map(map_size=(300, 300), random_seed=1999):
    train_set, test_set, valid_set = load_mnist()
    digit_data = np.vstack((train_set[0], valid_set[0], test_set[0]))
    digit_labels = np.concatenate((train_set[1], valid_set[1], test_set[1]))
    digit_size = (28, 28)
    random_state = np.random.RandomState(random_seed)
    mnist_map = np.zeros(map_size)
    for i in range(0, map_size[0], digit_size[0]):
        for j in range(0, map_size[1], digit_size[1]):
            which_sample = random_state.randint(0, digit_data.shape[0])
            digit = digit_data[which_sample].reshape(digit_size)
            digit = digit[:min(map_size[0] - i, 28), :]
            digit = digit[:, :min(map_size[1] - j, 28)]
            x_min = i
            x_max = i + digit_size[0]
            y_min = j
            y_max = j + digit_size[1]
            mnist_map[x_min:x_max, y_min:y_max] += digit
    mnist_map[mnist_map > 0] = 255.
    return mnist_map


def load_random_mnist_map(map_size=(300, 300), num_digits=50, random_seed=1999):
    train_set, test_set, valid_set = load_mnist()
    digit_data = np.vstack((train_set[0], valid_set[0], test_set[0]))
    digit_labels = np.concatenate((train_set[1], valid_set[1], test_set[1]))
    digit_size = (28, 28)
    random_state = np.random.RandomState(random_seed)
    mnist_map = np.zeros(map_size)
    for i in range(num_digits):
        which_sample = random_state.randint(0, digit_data.shape[0])
        digit = digit_data[which_sample].reshape(digit_size)
        # These are the centers
        x_pos = random_state.randint(0 + digit_size[0] // 2,
                                     map_size[0] - digit_size[0] // 2)
        y_pos = random_state.randint(0 + digit_size[1] // 2,
                                     map_size[1] - digit_size[1] // 2)
        x_min = x_pos - digit_size[0] // 2
        x_max = x_pos + digit_size[0] // 2
        y_min = y_pos - digit_size[1] // 2
        y_max = y_pos + digit_size[1] // 2
        mnist_map[x_min:x_max, y_min:y_max] += digit
    mnist_map[mnist_map > 0] = 255.
    return mnist_map
