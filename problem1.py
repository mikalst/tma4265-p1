# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from problem1plots import plot


def p_fixed_n(k, N):
    res = k / N
    for i in range(k + 1, N + 1):
        res += 1 / (i - 1)
    return res


def secretary_problem(k, candidates):
    """f(k, X) -> {b_1, b_2, b_3}
    
    Runs an instance of the secretary problem on the
    candidates, and returns an array with three boolean values: 
        -> b_1, Whether the algorithm successfully picks the largest value
        -> b_2, Whether the algorithm successfully picks a value
                among the top 3
        -> b_3, Whether the algorithm interviews all candidates
    """
    
    b_1 = 0
    b_2 = 0
    b_3 = 0
    
    # Sample the first k candidates
    x_star = max(candidates[0:k])
    
    # Run the algorithm
    x_chosen = None
    for j, cand in enumerate(candidates[k:-2]):
        if (cand > x_star):
            x_chosen = cand
            break
    # If we have not yet chosen a candidate we are forced to 
    # choose the last
    if (x_chosen == None):
        x_chosen = candidates[-1]
        b_3 = 1
    
    # Sort the candidate array and count backwards
    # until the chosen value is found
    candidates = np.sort(candidates)[::-1]
    n = 0
    for cand in candidates:
        n += 1
        if (cand == x_chosen):
            break    
    if n==1:
        b_1 = 1
    if n<=3:
        b_2 = 1
    
    return [b_1, b_2, b_3]


def simulate_const_n(k, N, iterations):
    """f(k, N, iterations) -> [r_best, r_top_three, r_interview_all]
    
    Runs iterations of the secretary problem with N candidates where the k first
    are observed. N can either be an fixed or an interval, in which N is chosen 
    randomly uniform in the interval. Returns an array with three floats
    between 0 and 1:
        -> r_best, the ratio of times the algorithm returned the best candidate
        -> r_top_three, the ratio of times the algorithm returned a
            candidate among the top three.
        -> r_interview_all, the ratio of times the algorithm interviewed all 
            candidates and subsequently chose the last candidate.
    """
    r_best = 0
    r_top_three = 0
    r_interview_all = 0
    
    # Checks wether we want a fixed or dynamic number of candidates
    fixed = True
    if not isinstance(N, int):
        fixed=False    

    for _ in range(iterations):
        
        # If N is not fixed, choose it randomly uniform in the interval
        if not fixed:
            N_candidates = np.random.randint(N[0], N[1])
        else:
            N_candidates = N
        
        candidates = np.random.randint(0, 100, N_candidates)
        result = secretary_problem(k, candidates)
        r_best += result[0]
        r_top_three += result[1]
        r_interview_all += result[2]
    
    r_best /= iterations
    r_top_three /= iterations
    r_interview_all /= iterations
        
    return [r_best, r_top_three, r_interview_all]
          

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

def main():
    print(simulate_const_n(11, 30, 1000))
    
    

if __name__ == "__main__":
    main()
    




