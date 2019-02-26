#!/usr/bin/env python

import math, sys
from gaussian_process import *
import numpy as np

l = float(sys.argv[1]) if len(sys.argv)>1 else 1.
u = float(sys.argv[2]) if len(sys.argv)>2 else 0.001
m = float(sys.argv[3]) if len(sys.argv)>3 else 1.

# https://www.librec.net/datagen.html
xs = [
4.3, 5.15, 7.55, 8.25, 10.55, 11.1, 12.25, 13.1, 13.6, 15., 17.05, 18.5, 20.4,
22.4, 24.6, 28.05, 31.4, 33.75, 35.4
]
ys = [
2.35, 4.4, 5., 2.95, 2.55, 5.1, 3.9, 6.9, 11.45, 10.25, 6.6, 10.4, 14.4, 11.25,
13.2, 13.75, 11.8, 14.65, 12.
]
us = [ u for x in xs ]
ts = np.linspace(0.,40.,1001).tolist()

def kernel(a, b):
    return m * math.exp(-0.5*(((a-b)/l)**2))

gp = gaussian_process(xs,ys,us,ts,kernel)

print 'regression complete'

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.figure(num=None, figsize=(6.5, 4), dpi=200)
plt.gca().fill_between(
    ts,
    [ m-u for m,u in gp ],
    [ m+u for m,u in gp ],
    color="#dddddd"
)
plt.plot(xs, ys, 'b.', ms=8)
plt.plot(ts, [ m for m,u in gp ], 'r-', lw=2)
# plt.axis([105,160,0,40])
plt.margins(x=0)

plt.savefig('test.pdf', bbox_inches='tight')

