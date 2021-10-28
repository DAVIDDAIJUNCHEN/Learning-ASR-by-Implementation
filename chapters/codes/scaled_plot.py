#!/usr/bin/env pyhon3

#import modules
import matplotlib.pyplot as plt
import numpy as np


ax = plt.subplot()
t1 = np.arange(0.0, 1.0, 0.01)

for n in [0.25, 0.5, 0.75, 1]:
    plt.plot(t1, t1**n, label=r"$\alpha$="+str(n))

leg = plt.legend(loc='lower right', ncol=1, mode="None", shadow=True, fancybox=True)
leg.get_frame().set_alpha(0.5)
plt.xlabel(r'$\mathbb{P}(S_{i}^{j})$')
h = plt.ylabel(r'$\mathbb{P}(S_{i}^{j})^{\alpha}$')
h.set_rotation(0)
plt.axvline(x=0.8, color='pink')
plt.axhline(y=0.8, color='red', ls='dotted')
plt.axhline(y=0.8**0.25, color='#1f77b4', ls='dotted')
plt.axvline(x=0.9, color='black')
plt.axhline(y=0.9, color='red', ls='dotted')
plt.axhline(y=0.9**0.25, color='#1f77b4', ls='dotted')
plt.plot([0.8], [0.8], 'r.', markersize=10.0)
plt.plot([0.9], [0.9], 'r.', markersize=10.0)
plt.plot([0.8], [0.8**0.25], color='#1f77b4', marker='.', markersize=10.0)
plt.plot([0.9], [0.9**0.25], color='#1f77b4', marker='.', markersize=10.0)


if __name__ == "__main__":
    plt.show()