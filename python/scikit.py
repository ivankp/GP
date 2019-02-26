#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# np.random.seed(1)

# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([
4.3, 5.15, 7.55, 8.25, 10.55, 11.1, 12.25, 13.1, 13.6, 15., 17.05, 18.5, 20.4,
22.4, 24.6, 28.05, 31.4, 33.75, 35.4
]).T

# Observations
# y = f(X).ravel()
y = [
2.35, 4.4, 5., 2.95, 2.55, 5.1, 3.9, 6.9, 11.45, 10.25, 6.6, 10.4, 14.4, 11.25,
13.2, 13.75, 11.8, 14.65, 12.
]

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, 40, 1001)).T

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(
    kernel = 1.*RBF(1, (1e-2, 1e2)) + WhiteKernel(1, (1e-10, 1e+1)),
    alpha = 0,
    n_restarts_optimizer = 0
)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

print gp
print gp.kernel_
hp = np.exp(gp.kernel_.theta)
print hp

y_pred, sigma = gp.predict(x, return_std=True)

plt.figure()
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - sigma, (y_pred + sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='stdev')
plt.margins(x=0)
plt.legend(loc='upper left')
plt.tight_layout()

plt.show()
