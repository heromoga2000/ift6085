from scipy.io import loadmat
import matplotlib.pyplot as plt
import numpy as np


def normalize_message(m):
    return m / np.sum(m)


def belief_propagation(transition_prob, visible_to_hidden_prob, observations):
    """
    My implementation
    Used reference notes from
    https://dl.dropboxusercontent.com/u/14115372/sum_product_algorithm/sum_product_note.pdf
    """
    n_observations = len(observations)
    # Forward pass
    forward = []
    # u f -> s from prev hidden
    hfac_message = np.array([0.8, 0.2])
    for i in range(n_observations):
        obs = observations[i]
        # Multiply the 2 factor messages, then marginalize over state i.e.
        # state i.e. multiply and sum (dot prod!)
        ofac_message = normalize_message(np.array([0.5, 0.5])
                                         * visible_to_hidden_prob[:, obs])
        forward_update = normalize_message(
            np.dot(transition_prob, ofac_message
                   * hfac_message))
        hfac_message = forward_update
        forward.append(forward_update)
    forward = np.array(forward)

    # Backward pass
    backward = []
    hfac_message = np.array([0.5, 0.5])
    for i in range(n_observations)[::-1]:
        obs = observations[i]
        ofac_message = normalize_message(np.array([0.5, 0.5])
                                         * visible_to_hidden_prob[:, obs])
        # Multiply the 2 factor messages, then marginalize over state i.e.
        # state i.e. multiply and sum (dot prod!)
        backward_update = normalize_message(
            np.dot(transition_prob, visible_to_hidden_prob[:, obs] * hfac_message))
        hfac_message = backward_update
        backward.insert(0, backward_update)
    backward = np.array(backward)
    marginal_prob = forward * backward
    marginal_prob /= np.sum(marginal_prob, axis=1)[:, None]
    return marginal_prob, forward, backward


# zt, zt+1
# Bad, Bad  | Bad, Good
# Good, Bad | Good, Good
transition_prob = np.array([[0.8, 0.2],
                            [0.2, 0.8]])

# xt, zt
# -1, Bad  | +1, Bad
# -1, Good | +1, Good
q = 0.7
visible_to_hidden_prob = np.array([[q, 1. - q],
                                   [1. - q, q]])

# Hardcoded in the function but will list it here as well
# x1
# Bad | Good
init_prob = np.array([[0.8, 0.2],])

# X is observed, Z is hidden state
X = loadmat('sp500.mat')['price_move']
X[ X > 0] = 1
X[ X < 0] = 0

pl, f, b = belief_propagation(transition_prob, visible_to_hidden_prob, X.ravel())

# xt, zt
# -1, Bad  | +1, Bad
# -1, Good | +1, Good
q = 0.9
visible_to_hidden_prob = np.array([[q, 1. - q],
                                   [1. - q, q]])
ph, f, b = belief_propagation(transition_prob, visible_to_hidden_prob, X.ravel())
plt.plot(pl[:, 1], label="q=0.7", color="steelblue")
plt.plot(ph[:, 1], label="q=0.9", color="darkred")
plt.title("Discrete HMM Belief Propagation")
plt.xlabel("Time (Weeks)")
plt.ylabel("Marginal probability of 'good' state")
plt.legend()
plt.show()
