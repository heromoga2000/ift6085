import numpy as np

from theano import config
from theano import tensor as T

from theano.compat.python2x import OrderedDict
from pylearn2.space import NullSpace
from pylearn2.train_extensions import TrainExtension
from pylearn2.utils import sharedX

from pylearn2.training_algorithms.learning_rule import LearningRule


class RMSprop_VD(LearningRule):
    """
    Implements the RMSprop

    Parameters
    ----------
    """

    def __init__(self,
                 init_momentum=.9,
                 averaging_coeff=.95,
                 stabilizer=.0001):
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        .. todo::

            WRITEME
        """
        # TODO: add channels worth monitoring
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        for param in grads.keys():

            #avg_grad = sharedX(np.zeros_like(param.get_value()))
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            momentum = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                #avg_grad.name = 'avg_grad_' + param.name
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            #new_avg_grad = self.averaging_coeff * avg_grad \
            #            + (1- self.averaging_coeff) * grads[param]
            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr \
                + (1 - self.averaging_coeff) * grads[param]**2

            #normalized_grad = grads[param] / T.sqrt(new_avg_grad_sqr \
            #                - new_avg_grad**2 + self.stabilizer)
            normalized_grad = grads[param] / T.sqrt(new_avg_grad_sqr
                                                    + self.stabilizer)
            new_momentum = self.momentum - learning_rate * normalized_grad

            #updates[avg_grad] = new_avg_grad
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[momentum] = new_momentum
            updates[param] = param + new_momentum

        return updates


class RMSprop(LearningRule):
    """
    Implements the RMSprop

    Parameters
    ----------
    """

    def __init__(self,
                 init_momentum=.9,
                 averaging_coeff=.95,
                 stabilizer=.0001):
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        .. todo::

            WRITEME
        """
        # TODO: add channels worth monitoring
        return

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()

        for param in grads.keys():

            inc = sharedX(param.get_value() * 0.)
            avg_grad = sharedX(np.zeros_like(param.get_value()))
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                avg_grad.name = 'avg_grad_' + param.name
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad = self.averaging_coeff * avg_grad \
                + (1 - self.averaging_coeff) * grads[param]
            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr \
                + (1 - self.averaging_coeff) * grads[param]**2

            normalized_grad = grads[param] / T.sqrt(new_avg_grad_sqr -
                                                    new_avg_grad**2 +
                                                    self.stabilizer)
            updated_inc = self.momentum * inc - learning_rate * normalized_grad

            updates[avg_grad] = new_avg_grad
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[inc] = updated_inc
            updates[param] = param + updated_inc

        return updates
