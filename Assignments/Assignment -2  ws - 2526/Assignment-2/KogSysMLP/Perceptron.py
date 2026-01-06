from __future__ import annotations

import numpy as np
from sklearn.base import ClassifierMixin


class Perceptron(ClassifierMixin):
    """
    Class implementing a sigmoid Perceptron
    """

    def __init__(
        self,
        ndim: int = 0,
    ):
        """
        Parameters
        ----------
        ndim: int (Optional)
            This parameter may be passed to set the dimensionality of the weights on construction
        """

        super().__init__()

        self.__w: np.ndarray = np.random.randn(ndim)
        self.__b: np.ndarray = np.random.randn(1)

    @property
    def w(self) -> np.ndarray:
        return self.__w

    @property
    def b(self) -> np.ndarray:
        return self.__b

    def update_weights(self, w: np.ndarray, b: np.ndarray) -> None:
        """
        A way to manually update the weight vector.

        Parameters
        ----------
        w: np.ndarray
            The new weight vector. Shape must match old weight vector.

        Raises
        ------
        ValueError
            If the shape of ``w`` does not match the shape of ``self.__w``
        """

        # ensure matching shapes
        if not w.shape == self.__w.shape:
            raise ValueError(
                f"Shape of new weight vector must match shape of old weight vector. Expected {self.__w.shape}, but got {w.shape}."
            )

        self.__w = w
        self.__b = b

    def __init_weights(self, X: np.ndarray) -> None:
        """
        Helper method to initialize weights. This is called from within fit if the dimensions of the input do not match the dimensions of the weights.

        Parameters
        ----------
        X: np.ndarray
            A training sample from which the correct dimensionality will be infered.
        """

        self.__w = np.random.randn(X.shape[-1])
        self.__b = np.random.randn(1)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the network output for input ``X``

        Parameters
        ----------
        X: np.ndarray
            Instances, either one-dimensional for a single instance or 2-dimensional for a set of instances.

        Returns
        -------
        np.ndarray
            Array of network outputs.
        """
        x = np.dot(X, self.__w) + self.__b
        x = 1 / (1 + np.exp(-x))

        return x

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.001,
        max_epoch: int = 500,
        resume: bool = False,
    ) -> Perceptron:
        """
        Train the Perceptron on ``X`` labeled with ``y``.

        Parameters
        ----------
        X: np.ndarray
            Training instances, either one-dimensional for a single instance or 2-dimensional for a set of instances.
        y: np.ndarrray
            Labels
        lr: float (Optional)
            Learning rate, default 0.001
        max_epoch: int (Optional)
            The maximum number of trianing epochs to run, default 500
        resume: bool (Optional)
            Resume training, i.e. do not initialize weights.

        Returns
        -------
        Perceptron
            itself

        Raises
        ------
        ValueError
            If ``X`` has an invalid number of dimensions.
        """

        # ensure valid inputs
        if X.ndim not in (1, 2):
            raise ValueError(
                f"Invalid Dimension: X is of dimension {X.ndim}, but must be of dimension 1 for a single instance or 2 for a set of instances."
            )

        # re-initialize weights, if necessary
        if X.shape[-1] != self.__w.shape[-0] and not resume:
            self.__init_weights(X)

        for epoch in range(max_epoch):
            for _x, _y in zip(X, y):
                self.__w += lr * (o := self.forward(_x)) * (1 - o) * (_y - o) * _x
                self.__b += lr * o * (1 - o) * (_y - o)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Turn the netweok output into a concept prediction of 1 or 0.

        Parameters
        ----------
        X: np.ndarray
            Instances, either one-dimensional for a single instance or 2-dimensional for a set of instances.

        Returns
        -------
        np.ndarray
            Array of concept predictions.
        """
        return np.array([1 if o > 0.5 else 0 for o in self.forward(X)])
