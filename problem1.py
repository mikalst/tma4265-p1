# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np


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
    
    # if k > N one will always interview all candidates with no chance
    if k > len(candidates):
        return [0, 0, 1]
    
    # Sample the first k candidates
    x_star = max(candidates[0:k])
    
    # If we find a better candidate among the k+1...n, choose it
    x_chosen = None
    for cand in candidates[k:-1]:
        if (cand > x_star):
            x_chosen = cand
            break
    # If interviewed all but the last candidate, we are forced to choose it and
    # will thus have interviewed all candidates
    if (x_chosen == None):
        x_chosen = candidates[-1]
        b_3 = 1
    
    # Sort the candidate array
    candidates = np.sort(candidates)
        
    # Check if chosen value is best or top 3
    if (x_chosen == candidates[-1]):
        b_1 = 1
    if (x_chosen in candidates[-3:]):
        b_2 = 1
    
    return [b_1, b_2, b_3]


def simulate(k, N, iterations):
    """f(k, N, iterations) -> [r_best, r_top_three, r_interview_all]
    
    Runs iterations of the secretary problem with N candidates where the k
    first are observed. N can either be an fixed or an interval, in which N is
    chosen randomly uniform in the interval. Returns an array with three floats
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
        
        # If N is not fixed, choose it randomly uniform in the closed interval
        if not fixed:
            N_candidates = np.random.randint(N[0], N[1]+1)
        else:
            N_candidates = N
        
        candidates = np.random.randint(0, 1e9, N_candidates)
        result = secretary_problem(k, candidates)
        r_best += result[0]
        r_top_three += result[1]
        r_interview_all += result[2]
    
    r_best /= iterations
    r_top_three /= iterations
    r_interview_all /= iterations
        
    return [r_best, r_top_three, r_interview_all]
          

def p_fixed_n(k, N):
    res = 0
    for i in range(k+1, N + 1):
        res += 1 / (i - 1)
    res *= k / N
    return res


def p_varying_n(k):
    res = 0
    for N in range(16, 45 + 1):
        res += p_fixed_n(k, N)
    res /= 30
    return res


def plot_analytical():
    k_1 = range(1, 30)
    k_2 = range(1, 40)
    y_1 = [p_fixed_n(k, 30) for k in k_1]
    y_2 = [p_fixed_n(k, 40) for k in k_2]
    
    plt.title("Analytical with fixed $N$, $N \in \{30, 40\}$")
    plt.style.use('ggplot')
    plt.plot(k_1, y_1)
    plt.plot(k_2, y_2)
    plt.plot([11, 11], [0, y_1[11]])
    plt.plot([15, 15], [0, y_2[15]])
    plt.xlabel("$k$")
    plt.ylabel("$P(Z=1|K=k)$")
    plt.legend(["fixed $N=30$", "fixed $N=40$", "$x=11$", "$x=16$"])
    plt.savefig("oppg1b")
    
    
def plot_stochastic_n():
    k_values = range(1, 45)
    y_1 = [p_varying_n(k) for k in k_values]
    
    plt.title("Analytical with stochastic $N$, $N \in [16, 45]$")
    plt.style.use('ggplot')
    plt.plot(k_values, y_1)
    plt.plot([9,9],[0, y_1[9]-0.001])
    plt.xlabel("$k$")
    plt.ylabel("$P(Z=1|K=k)$")
    plt.legend(["stochastic $N \in [16, 45]$", "$x=10$"])
    plt.savefig("oppg1d")



def main():
    flag = {0: "1b",
            1: "1c",
            2: "1d"}[2]
    
    if flag == "1b":
        plot_analytical()
        
    elif flag == "1c":
        for k in range(9, 14):
            print(simulate(k, 30, 100000))
            print(p_fixed_n(k, 30))
        
    elif flag == "1d":
        plot_stochastic_n()
        for k in range(8, 13):
            print(simulate(k, [16, 45], 100000))
            print(p_varying_n(k))
    

if __name__ == "__main__":
    main()
    




