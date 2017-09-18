# -*- coding: utf-8 -*-
import numpy as np
from problem1plots import plot1, plot2

def p_fixed_n(n, k):
    #Probability of choosing the best candidate with 
    #N fixed = n
    res = 0
    for i in range(k + 1, n + 1):
        res += 1 / (i - 1)
    res *= (k / n)
    return res

def p(k):
    #Probability of choosing the best candidate with N 
    #uniformly distributed in { 16, 17, ..., 45 }
    res = 0
    for i in range(16, 46):
        res += p_fixed_n(i, k);
    res /= 30            
    return res

def simulate(k):
    #A random integer between 16 and 45 is chosen uniformly
    n = round(np.random.uniform(15.5, 45.49))
    choise = n - 1
    #list holding the top three candidates
    best = [0, 0, 0]
    #the best candidate among the first k
    best_k = 0
    candidates = []
    for i in range(0, n):
        #a candidate's value is uniformly distributed between
        #1 and 1000
        c = np.random.uniform(1, 1000)
        candidates.append(c)
        if (best[0] <= c):
            best[2] = best[1]
            best[1] = best[0]
            best[0] = c
        if i < k:
            best_k = max(best_k, c)
            
    for i in range(k, n):
        if candidates[i] > best_k:
            #the candidate is finally chosen
            #accoring to the algorithm
            choise = i
            break
        
    if candidates[choise] == best[0]:
        return 1
    else:
        return 0
    
def run_simulations(n, k):
    successes = 0
    for i in range(n):
        successes += simulate(k)
    return successes / n

def main():
    
    #plotting in b)
    kvalues1 = [ i for i in range(1, 31) ]
    pvalues1 = [ p_fixed_n(30, k) for k in kvalues1 ]
    kvalues2 = [ i for i in range(1, 41) ]
    pvalues2 = [ p_fixed_n(40, k) for k in kvalues2 ]
    plot2(kvalues1, pvalues1, kvalues2, pvalues2, "$P(Z = 1)$ for fixed $N = 30, 40$",
         "$k$", "$p$", 30 / np.e, 40 / np.e, 1, 40, 0, 0.5)
    
    #plotting in d)
    kvalues = [ i for i in range(1, 16) ]
    pvalues = [ p(k) for k in kvalues ]
    plot1(kvalues, pvalues, "$P(Z = 1)$ with uniformly distributed candidate size", "$k$", "$p$",
         10, 1, 15, 0, 0.5)
    
    #simulate successes in chosing the best candidate in task d)
    result = run_simulations(10000, 10)
    print(result)
    #a run gives 0.36265, which is very close to the actual probabily, 0.3617


if __name__ == "__main__":
    main()
    




