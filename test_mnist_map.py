from mnist_map import load_random_mnist_map, load_orderly_mnist_map
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

random_mnist_map = load_random_mnist_map()
plt.matshow(random_mnist_map, cmap="gray")
plt.savefig('/u/kastner/random_mnist_map.png')

orderly_mnist_map = load_orderly_mnist_map()
plt.matshow(orderly_mnist_map, cmap="gray")
plt.savefig('/u/kastner/orderly_mnist_map.png')
