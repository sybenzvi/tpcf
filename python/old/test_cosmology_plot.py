# coding: utf-8
from cosmology import Cosmology
import numpy as np
import matplotlib.pyplot as plt
c = Cosmology()
z = np.linspace(0, 1, 10001)

fig, ax = plt.subplots(1,1, figsize=(8,6))
ax.plot(c.z, c.r)
ax.plot(z, c.z2r(z), '.')
ax.set(xlabel='redshift $z$',
       xlim=[0,1],
       ylabel='comoving distance $r$ [$h^{-1}$ Mpc]')

fig.tight_layout()

plt.show()
