# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from problem1plots import plot

N = 30

def p_fixed_n(k):
    res = 0
    for i in range(k + 1, N + 1):
        res += 1 / (i - 1)
    res *= (k / N)
    return res

def simulate(k):
    N = round(np.random.uniform(16, 45))
    choise = N - 1
    best = [0, 0, 0]
    best_k = 0
    candidates = []
    for i in range(0, N):
        c = np.random.uniform(1, 1000)
        candidates.append(c)
        if (best[0] <= c):
            best[2] = best[1]
            best[1] = best[0]
            best[0] = c
        if i < k:
            best_k = max(best_k, c)
    for i in range(k, N):
        if candidates[i] > best_k:
            choise = i
            break
    if candidates[choise] == best[0]:
        return 1
    else:
        return 0

def p(k):
    res = 0
    for i in range(16, 46):
        _res = 0
        for j in range(k + 1, i + 1):
            _res += 1 / (j - 1)
            
        _res *= k / i
        res += _res
    res /= 30            
    return res

<<<<<<< HEAD

def main():
    """
    kvalues = [ i for i in range(1, 30 + 1) ]
    pvalues = [ p_fixed_n(k) for k in kvalues ]
    plot(kvalues, pvalues, "Probabilty of choosing best candidate for fixed N = 30",
         "$k$", "$P(Z = 1 | K = k)$", 30 / np.e, 1, 31, 0, 0.5)
    """
    
    kvalues = [ i for i in range(1, 16) ]
    pvalues = [ p(k) for k in kvalues ]
    plot(kvalues, pvalues, "$P(Z = 1)$, given uniformly distrbuten candidate size", "$k$", "$p$",
         10, 1, 15, 0, 0.5)
    
=======
def main(n_sim):
    
    kvalues = [ i for i in range(1, 16) ]
    pvalues = [ p(k) for k in kvalues ]

    plt.figure(1)
    plt.plot(kvalues, pvalues)
    plt.show()

>>>>>>> 9c90c730c7acf1d82381f453c37220233abd0eca
    return 0

if __name__ == "__main__":
    main(10)
    




