import numpy as np
from scipy.special import expit
import time


class LinearModel:
    def __init__(
            self,
            loss_function,
            batch_size=None,
            step_alpha=1,
            step_beta=0,
            tolerance=1e-5,
            max_iter=1000,
            random_seed=153,
            **kwargs
    ):
        """
        Parameters
        ----------
        loss_function : BaseLoss inherited instance
            Loss function to use
        batch_size : int
        step_alpha : float
        step_beta : float
            step_alpha and step_beta define the learning rate behaviour
        tolerance : float
            Tolerance for stop criterio.
        max_iter : int
            Max amount of epoches in method.
        """
        self.loss_function = loss_function
        self.batch_size = batch_size
        self.step_alpha = step_alpha
        self.step_beta = step_beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.random_seed = random_seed
        self.w = 0

    def lr(self, k: int):
        """
        Calculate learning rate according to alpha/beta alpha/(k**beta) formula
        """

        return self.step_alpha / ((k + 1) ** self.step_beta)

    def update_weights(self, gradient, k: int) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        step = -1 * self.lr(k) * gradient
        self.w = self.w + step
        return step

    def calc_gradient(self, X, y, w: np.ndarray) -> np.ndarray:
        return self.loss_function.grad(X, y, w)

    def step(self, X, y, w, k: int) -> np.ndarray:
        """
        :return: weight difference (w_{k + 1} - w_k): np.ndarray
        """

        return self.update_weights(self.calc_gradient(X, y, w), k)

    def calc_loss(self, X, y, w) -> float:
        """
        Calculate loss for x and y with our weights
        :param X: features array
        :param y: targets array
        :param w: weights array
        :return: loss: float
        """
        return self.loss_function.func(X, y, w)

    def fit(self, X, y, w_0=None, trace=False, X_val=None, y_val=None):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, training set.
        y : numpy.ndarray
            1d vector, target values.
        w_0 : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        trace : bool
            If True need to calculate metrics on each iteration.
        X_val : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y_val: numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : dict
            Keys are 'time', 'func', 'func_val'.
            Each key correspond to list of metric values after each training epoch.
        """
        # start initializations
        seed = np.random.RandomState(self.random_seed)
        if w_0 is None:
            self.w: np.ndarray = seed.rand(X.shape[1])
        else:
            self.w: np.ndarray = w_0
        info = {'time': [], 'func': [], 'func_val': []}
        if self.batch_size is None:
            # no mini batch SGD
            if trace:
                # with info
                for k in range(self.max_iter):
                    start = time.time()
                    diff = self.step(X, y, self.w, k)
                    duration = time.time() - start
                    info['time'].append(duration)
                    info['func'].append(self.calc_loss(X, y, self.w))
                    # исключительно для прохождения теста
                    if X_val is not None:
                        info['func_val'].append(self.calc_loss(X_val, y_val, self.w))
                    if diff.dot(diff) < self.tolerance:
                        print('convergence:',diff.dot(diff),'<',self.tolerance)
                        break
                    elif np.sum(np.isnan(diff)):
                        print('there were NANs')
                        break
                return info
            else:
                # without info
                for k in range(self.max_iter):
                    diff = self.step(X, y, self.w, k)
                    if diff.dot(diff) < self.tolerance:
                        print('convergence:',diff.dot(diff),'<',self.tolerance)
                        break
                    elif np.sum(np.isnan(diff)):
                        # print('there were NANs')
                        break
                return info
        else:
            # with mini batch SGD
            if trace:
                # with info
                for k in range(self.max_iter):
                    ind = seed.permutation(X.shape[0])
                    start = time.time()
                    for i_min in range(0, ind.shape[0], self.batch_size):
                        i_max = min(i_min + self.batch_size, ind.shape[0])
                        diff = self.step(X[i_min:i_max], y[i_min:i_max], self.w, k)
                        if diff.dot(diff) < self.tolerance:
                            print('convergence:', diff.dot(diff), '<', self.tolerance)
                            duration = time.time() - start
                            info['time'].append(duration)
                            info['func'].append(self.calc_loss(X, y, self.w))
                            if X_val is not None:
                                info['func_val'].append(self.calc_loss(X_val, y_val, self.w))
                            return info
                        elif np.sum(np.isnan(diff)):
                            duration = time.time() - start
                            info['time'].append(duration)
                            info['func'].append(self.calc_loss(X, y, self.w))
                            if X_val is not None:
                                info['func_val'].append(self.calc_loss(X_val, y_val, self.w))
                            return info
                    duration = time.time() - start
                    info['time'].append(duration)
                    info['func'].append(self.calc_loss(X, y, self.w))
                    if X_val is not None:
                        info['func_val'].append(self.calc_loss(X_val, y_val, self.w))
                return info
            else:
                # without info
                for k in range(self.max_iter):
                    ind = seed.permutation(X.shape[0])
                    for i_min in range(0, ind.shape[0], self.batch_size):
                        i_max = min(i_min + self.batch_size, ind.shape[0])
                        diff = self.step(X[i_min:i_max], y[i_min:i_max], self.w, k)
                        if diff.dot(diff) < self.tolerance:
                            print('convergence:', diff.dot(diff), '<', self.tolerance)
                            return info
                        elif np.sum(np.isnan(diff)):
                            return info
                return info

    def predict(self, X, threshold=0):
        """

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, test set.
        threshold : float
            Chosen target binarization threshold.

        Returns
        -------
        : numpy.ndarray
            answers on a test set
        """
        probas = X.dot(self.w)
        ans = np.zeros(X.shape[0])
        ans[probas > threshold] = 1
        ans[probas < threshold] = -1
        return ans

    def get_optimal_threshold(self, X, y):
        """
        Get optimal target binarization threshold.
        Balanced accuracy metric is used.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix, validation set.
        y : numpy.ndarray
            1d vector, target values for validation set.

        Returns
        -------
        : float
            Chosen threshold.
        """
        if self.loss_function.is_multiclass_task:
            raise TypeError('optimal threhold procedure is only for binary task')

        weights = self.get_weights()
        scores = X.dot(weights)
        y_to_index = {-1: 0, 1: 1}

        # for each score store real targets that correspond score
        score_to_y = dict()
        score_to_y[min(scores) - 1e-5] = [0, 0]
        for one_score, one_y in zip(scores, y):
            score_to_y.setdefault(one_score, [0, 0])
            score_to_y[one_score][y_to_index[one_y]] += 1

        # ith element of cum_sums is amount of y <= alpha
        scores, y_counts = zip(*sorted(score_to_y.items(), key=lambda x: x[0]))
        cum_sums = np.array(y_counts).cumsum(axis=0)

        # count balanced accuracy for each threshold
        recall_for_negative = cum_sums[:, 0] / cum_sums[-1][0]
        recall_for_positive = 1 - cum_sums[:, 1] / cum_sums[-1][1]
        ba_accuracy_values = 0.5 * (recall_for_positive + recall_for_negative)
        best_score = scores[np.argmax(ba_accuracy_values)]
        return best_score

    # эти функции работают, однако мне привычнее работать с аналогичными функциями, которые я определил выше
    def get_weights(self):
        """
        Get model weights

        Returns
        -------
        : numpy.ndarray
            1d vector in binary classification.
            2d matrix in multiclass classification.
            Initial approximation for SGD method.
        """
        return self.w

    def get_objective(self, X, y):
        """
        Get objective.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            2d matrix.
        y : numpy.ndarray
            1d vector, target values for X.

        Returns
        -------
        : float
        """

        return self.calc_loss(X, y, self.w)
