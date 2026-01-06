from __future__ import annotations
from numpy.typing import ArrayLike
from copy import copy

import numpy as np
from sklearn.base import ClassifierMixin

from Perceptron import Perceptron


class MLP(ClassifierMixin):
    """ """

    __layers: ArrayLike
    __matrices: list[tuple]
    __classes: ArrayLike

    def __init__(self, hidden_layers: list[int] = [50]) -> None:
        """
        Parameters
        ----------
        hidden_layers: list[int]
            Number of Perceptrons in each hidden layer.

        Notes
        -----
        - To reduce complexity, layers aren't initialized in the constructor, but during training when the number of classes and input features are known.
        """

        self.__hidden: list[int] = hidden_layers

    @property
    def hidden(self) -> list[int]:
        return self.__hidden

    @property
    def layers(self) -> ArrayLike:
        return self.__layers

    @property
    def matrices(self) -> list[tuple]:
        return self.__matrices

    @property
    def classes(self) -> ArrayLike:
        return self.__classes

    def __init_layers(self, X: ArrayLike, y: ArrayLike) -> None:
        """
        Initialize layers of Perceptrons based on the number of features in ``X`` and the number of unique classes in ``y``.

        Parameters
        ----------
        X: ArrayLike
            A training sample from which the correct input dimensionality to the first layer will be infered.
        y: ArrayLike
            Labels for the training samples, from which the number of classes will be infered.

        Note
        ----
        - This approach helps with the intuition of an MLP being Perceptrons with links. However, this implementation will be quite slow, as the only way to apply each Perceptron is a for-loop (all numpy options here are also just wrappers for for-loops).
        - An alternate way of doing this is to not use Perceptrons at all, but represent each hidden layer as a weight matrix of dimensions (n_percs, w_dim) and an additional bias vector of dimension (n_percs,).
        - Finally, we can keep the individual perceptron representation and extract the weights and biases into the aforementioned matrices whenever they are needed, perform calculations quickly, and then store the weights in the individual Perceptrons.
        """

        # set known classes vector based on y
        self.__classes = np.unique(y)

        # build construct_layers by extending hidden with the number of unique classes
        construct_layers = copy(self.__hidden)
        construct_layers.append(len(np.unique(y)))

        # initialize layer array
        self.__layers = np.zeros((len(construct_layers),), dtype=object)

        # fill layer array with perceptrons
        n_in = X.shape[-1]
        for layer, n in enumerate(construct_layers):
            self.__layers[layer] = np.array([Perceptron(n_in) for _ in range(n)])
            n_in = n

        self.update_matrices()

    def update_matrices(self) -> None:
        """
        Turn perceptron objects into a matrix representation for fast computation.
        """
        tmp = []
        for layer in self.__layers:
            tmp.append((
                np.array([p.w for p in layer]),
                np.array([p.b[0] for p in layer]),
            ))

        self.__matrices = tmp

    def __one_hot(self, y: ArrayLike) -> ArrayLike:
        """
        Turn label-vector into one-hot vector.

        Parameters
        ----------
        y: ArrayLike
            Original label vector

        Returns
        -------
        ArrayLike
            Labels as one-hot vectors
        """
        # check if y is an array
        y = np.array([y]) if not isinstance(y, ArrayLike) else y

        if y.ndim == 2:
            return y

        Y = np.zeros((len(y), len(self.__classes)))
        Y[
            np.arange(len(y)),
            np.array(list(map(lambda x: list(self.__classes).index(x), y))),
        ] = 1
        return Y

    def __forward(self, X: ArrayLike) -> list[ArrayLike]:
        """
        Calculate detailed forward pass for backpropagation.

        Parameters
        ----------
        X: ArrayLike
            Network input

        Returns
        -------
        list[ArrayLike]
            List containing the original inputs and outputs of all layers in order.
        """
        activations = [X]
        a = X
        for w, b in self.__matrices:
            a = self.process_one_layer(a, w, b)
            activations.append(a)
        return activations

    def backprop(self, X: ArrayLike, y: ArrayLike, lr: float) -> None:
        """
        Single run of backpropagation inner loop, can be called with either batches

        Parameters
        ----------
        X: ArrayLike
            Single sample or batch sample
        y: ArrayLike
            Corresponding labels
        lr: float
            Learning rate
        """
        # 24 Points
        # TODO: implement method
        raise NotImplementedError

    def fit(
            self,
            X: ArrayLike,
            y: ArrayLike,
            lr: float = 0.01,
            max_epoch: int = 200,
            resume: bool = False,
    ) -> MLP:
        """
        Train the MLP on ``X`` labeled with ``y``.

        Parameters
        ----------
        X: ArrayLike
            Training instances, either one-dimensional for a single instance or 2-dimensional for a set of instances.
        y: ArrayLike
            Labels
        lr: float (Optional)
            Learning rate, default 0.01
        max_epoch: int (Optional)
            The maximum number of training epochs to run, default 200
        resume: bool (Optional)
            Resume training, i.e. do not initialize weights.

        Returns
        -------
        MLP
            itself
        """
        # 7 Points
        # TODO: implement method
        raise NotImplementedError

    def forward(self, X: ArrayLike) -> ArrayLike:
        """
        Calculate network output by sequentially calculating layer outputs.

        Parameters
        ----------
        X: ArrayLike
            Input features

        Returns
        -------
        ArrayLike
            Output of model function
        """
        for w, b in self.__matrices:
            X = self.process_one_layer(X, w, b)

        return X

    def predict(self, X: ArrayLike) -> ArrayLike:
        """
        Turn network output into class prediction.

        Parameters
        ----------
        X: ArrayLike
            Input features

        Returns
        -------
        ArrayLike
            Class predictions for each input instance
        """
        return np.array([
            self.__classes[pred] for pred in self.forward(X).argmax(axis=1)
        ])

    @staticmethod
    def process_one_layer(
            input: ArrayLike, layer_w: ArrayLike, layer_b: ArrayLike
    ) -> ArrayLike:
        """
        Efficiently calculate layer output using matrix multiplication.

        Parameters
        ----------
        input: ArrayLike
            The input to the layer
        layer_w: ArrayLike
            The weight matrix of the layer
        laber_b: ArrayLike
            The bias matrix of the layer

        Returns
        -------
        ArrayLike
            input · layer_w + layer_b
        """
        x = (input @ layer_w.T) + layer_b
        return 1 / (1 + np.exp(-x))
