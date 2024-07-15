import pandas as pd
import numpy as np
import random
from typing import Union

class MyLineReg():
    def __init__(self, 
                n_iter: int = 100,
                learning_rate: float = 0.1,
                metric: str = None,
                reg: str = None,
                l1_coef: float = 0.0,
                l2_coef: float = 0.0,
                sgd_sample: Union[int, float] = None,
                random_seed: int = 42
                ) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.metric = metric
        self.__weights: np.array = None
        self.__best_score = None
        self._reg = reg
        self._l1_coef = l1_coef
        self._l2_coef = l2_coef
        self._sgd_sample = sgd_sample
        self._random_seed = random_seed
    
    def __str__(self) -> str:
        params = [f'{key}={value}' for key, value in self.__dict__.items()]
        return f'{self.__class__.__qualname__} class: ' + ', '.join(params)

    def fit(self, 
            X: pd.DataFrame,
            y: pd.Series,
            verbose: int = False
            ) -> None:
        
        random.seed(self._random_seed)

        n, num_features = X.shape
        X, y = X.to_numpy(), y.to_numpy()
        X = np.hstack(([[1]] * n, X))
        self.__weights = np.ones(num_features + 1)

        verbose_str = ''

        for i in range(self.n_iter):
            if isinstance(self._sgd_sample, int):
                sample_rows_idx = random.sample(range(X.shape[0]), self._sgd_sample)
            elif isinstance(self._sgd_sample, float):
                sample_rows_idx = random.sample(range(X.shape[0]), round(self._sgd_sample * n))
            else:
                sample_rows_idx = list(range(X.shape[0]))
            
            batch_x = X[sample_rows_idx]
            batch_y = y[sample_rows_idx]

            y_pred = X @ self.__weights 
            loss = self.__get_loss(y, y_pred)
            grad = self.__get_gradient(batch_x, batch_y)

            if isinstance(self.learning_rate, float):
                self.__weights -= self.learning_rate * grad
            elif callable(self.learning_rate):
                self.__weights -= self.learning_rate(i + 1) * grad

            if self.metric != None:
                self.__best_score = getattr(self, f'_{self.metric}')(y, X @ self.__weights)
                verbose_str = f' | {self.metric}: {self.__best_score}'

            if verbose and i % verbose == 0:
                    print(f'{i} | loss: {loss} {verbose_str}')
       

    def get_coef(self) -> np.array:
        return np.array(self.__weights)[1:]
    
    def predict(self,
                X: pd.DataFrame
                )-> np.array:
        X = X.to_numpy()
        X = np.hstack((np.ones(len(X)), X))
        return X @ self.__weights
    
        
    def get_best_score(self) -> float:
        return self.__best_score

    def __get_loss(self, y: np.array, y_pred: np.array) -> float:
        loss = sum((y - y_pred)**2) / len(y)
        if self._reg:
            loss += self._l1_coef * self.__l1() + self._l2_coef * self.__l2()
        return loss
    
    def __get_gradient(self, batch_x: np.array, 
                 batch_y: np.array) -> np.array:
        grad = 2 * ((batch_x @ self.__weights - batch_y) @ batch_x) / len(batch_x)
        if self._reg:
            grad += self._l1_coef * self.__l1_grad() + self._l2_coef * self.__l2_grad()
        return grad
    
    def __l1(self):
        return sum(abs(self.__weights))
    def __l2(self):
        return sum(self.__weights ** 2)

    def __l1_grad(self):
        return np.sign(self.__weights)
    def __l2_grad(self):
        return 2 * self.__weights

    @staticmethod
    def _mae(y: np.array, y_pred: np.array) -> float:
        return sum(abs(y - y_pred)) / len(y)
    @staticmethod
    def _mse(y:np.array, y_pred: np.array) -> float:
        return sum((y - y_pred)**2)/ len(y)
    @staticmethod
    def _rmse(y: np.array, y_pred: np.array) -> float:
        return np.sqrt(sum((y - y_pred)**2) / len(y))
    @staticmethod
    def _r2(y: np.array, y_pred: np.array) -> float:
        return 1 - (sum((y - y_pred)**2))/(sum((y-np.mean(y))**2))
    @staticmethod
    def _mape(y: np.array, y_pred: np.array) -> float:
        return 100 * sum(abs((y-y_pred) / y)) / len(y)