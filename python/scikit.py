#!/usr/bin/env python

import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

# np.random.seed(1)

# ----------------------------------------------------------------------
#  First the noiseless case
X = np.atleast_2d([
-7.288352, -6.354489, -6.343246, -5.873644, -4.744751, -4.092581, -3.743588,
-2.788551, -2.173762, -0.897309, 0.500722, 0.770547, 1.009111, 2.386,
2.475254, 4.245013, 4.332956, 4.908614, 5.837199, 6.143407
]).T

# Observations
# y = f(X).ravel()
y = [
-1.753582, -0.048711, 0.030086, 0.244986, -0.815186, -1.223496, -1.173352,
0.388252, 1.405444, 1.691977, -0.13467, -0.743553, -1.022923, -2.69914,
-2.226361, -1.287966, -1.116046, -1.574499, -1.080229, -0.84384
]

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(X[0], X[-1], 1001)).T

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(
    kernel = 1.08*RBF(0.3, (1e-2, 1e2)) + WhiteKernel(5e-5, (1e-10, 1e+1)),
    # optimizer = None
    optimizer = 'fmin_l_bfgs_b'
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
