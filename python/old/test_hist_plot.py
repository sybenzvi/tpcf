from hist_fast import Hist1D, Hist2D
import numpy as np
import matplotlib.pyplot as plt

hx = Hist1D(20, -4., 4.)
hy = Hist1D(20, -4., 4.)
h2 = Hist2D(20, -4., 4., 20, -4., 4.)

for x, y in np.random.multivariate_normal([-1,2], [[1,0.5], [0.5,2]], 100000):
    hx.fill(x)
    hy.fill(y)
    h2.fill(x, y)

fig, ax = plt.subplots(2,2, figsize=(8,6))
[axx, axy], [ax2, ax2e] = ax

edges = hx.xedges[:-1]
width = hx.xedges[1:] - hx.xedges[:-1]
axx.bar(edges, hx.data_w, width)
axx.set(xlabel='$x$', ylabel='count')

edges = hy.xedges[:-1]
width = hy.xedges[1:] - hy.xedges[:-1]
axy.bar(edges, hy.data_w, width)
axy.set(xlabel='$y$', ylabel='count')

img = ax2.imshow(np.flipud(h2.getValue()), interpolation='nearest',
                 extent=[h2.xmin, h2.xmax, h2.ymin, h2.ymax])
ax2.set(title='Values', xlabel='$x$', ylabel='$y$')
fig.colorbar(img, ax=ax2, label='count')

img = ax2e.imshow(np.sqrt(np.flipud(h2.getValue())), interpolation='nearest',
                extent=[h2.xmin, h2.xmax, h2.ymin, h2.ymax])
ax2e.set(title='Uncertainties', xlabel='$x$', ylabel='$y$')
cb = fig.colorbar(img, ax=ax2e, label='count')

fig.tight_layout()
plt.show()
