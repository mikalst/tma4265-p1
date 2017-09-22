import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('ggplot') #to be used for pthe plot in task a

#Making transition matrix
P = np.matrix([[0.95, 0.05], [0, 1]])
#initialized vector
x = np.matrix([[0.99, 0.01]])


def marginal_prob(k):
    if (k == 1):
        return 0.99
    else:
        return 0.99 * 0.95**(k-1)

#a
def task_a(P, x):
    '''
    Doing 50 iterations and storing the probability for class 2
    '''
    #list to append probabilities and appending the first probability
    prob_1 = [0.99]
    prob_2 = [0.01]
    
    

    #doing 50 iterations
    for k in range (1, 50):
        v = x*(P**k)
        prob_1.append(v[0, 0])
        prob_2.append(v[0, 1])
        
    return prob_1, prob_2

def plot_a(P,x):
    '''
    plotting the probability as a funtion of n
    '''
    probabilities = task_a(P, x)
    #plotting marginal probabilities
    plt.figure()
    plt.plot(range(1,51), probabilities[::2]) #flattened list
    plt.title('Marginal probabilities for $P(X_n = 1)$ as function of $n$.', fontsize=14)
    plt.xlabel('$n$')
    plt.ylabel('$P(X_n = 1)$')
    plt.show()

#plot_a(P,x)

#b
def task_b_simulate_hill(n_sim, P, x):
    '''
    The problem is identical to a geometric distribution
    and is programmed in that way. A list of markov chains is returned
    '''
    #list to store each simulation
    realizations = []
    count = 0 # debugging
    #simulating
    for _ in range(n_sim):
        #setting all roads to risk 1
        x = np.ones((50,), dtype=np.int)
        
        #simulating each road
        for i in range(1,51):
            if i == 1:  #initial probability for i = 1
                prob = 0.99
            else:        #probability is 0.95 if i is not 1
                prob = 0.95
            #drawing from a uniform distribution
            random = np.random.uniform(0,1)
            if random > prob:
                #if there is a road of class 2 every road above is set to risk 2
                for n in range(50-i):
                    x[i+n] = 2
                break

        realizations.append(x)

    mean = np.zeros(50)
    for r in realizations:
        for i, state in enumerate(r):
            if state==1:
                mean[i]+=1

    means = np.divide(mean, n_sim)
    
    #returning the list of the simulations
    return realizations, means

#task_b_simulate_hill(50, P, x)

def plot_b(P, x):
    '''
    plotting the realizations from task 2b with imshow
    '''
    image, means = task_b_simulate_hill(25, P, x)

    plt.figure()
    ax = plt.gca() #to size the colorbar
    im = ax.imshow(image)
    plt.title('$25$ realizations as a function of time')
    plt.xlabel('time')
    plt.ylabel('Realization number')
    #more stuff for the colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    ax.set_ylim(ax.get_ylim()[::-1])
    #må få til å slutte aksene på siste verdi
    plt.show()

    plt.figure()
    plt.title('Average result of the realizations as a function of $n$.')
    plt.plot(range(1,51), means)
    plt.xlabel('$n$')
    plt.ylabel('Average')
    plt.show()

#plot_b(P, x)

def task_c(P, x, k):
    '''
    computing forward and backward probability
    '''
    fw = np.matrix([[0.95, 0.05], [0, 1]])
    bw = np.matrix([[1, 0], [0.05, 0.95]])

    v1 = [1,0]
    v2 = [0,1]
    y1 = []
    y2 = []

    for l in range(k-1,0, -1):
        y1.append((v1*bw**l)[0,1])
        y2.append((v2*bw**l)[0,1])

    for l in range(0, 51-k):
        y1.append((v1*fw**l)[0,1])
        y2.append((v2*fw**l)[0,1])

    return y1, y2

def plot_c(P, x, k):
    y1, y2 = task_c(P, x, k)
    print('Length og y1 and y2: ', len(y1), ', ', len(y2))
    plt.style.use("ggplot")
    plt.title('Forward and backward propabilities as a function of $l$')
    plt.ylabel("$P(X_l = 2)$")
    plt.xlabel("$l$")
    plt.plot(range(1,51), y1)
    plt.plot(range(1,51), y2)
    plt.legend(["$P(X_l = 2 | X_{20} = 1)$", "$P(X_l = 2 | X_{20} = 2)$"])
    plt.show()


def task_d(P, x):
    '''
    This function gets the probabilities from task a 
    and sums over these with a cost of 5000 per road.
    '''
    marg_1, marg_2 = task_a(P, x)
    #print(probabilities)
    price = 0
    count = 0
    for prob in marg_2:
        #print(prob)
        price+=5000*prob
        count+=1
    print('\ntotal cost = ', price)
    print('optimal choice is: ')
    if price < 100000:
        print('5000 per road ')
        return price
    else:
        print('fixed price of 100000')
        return 100000

#task_d(P, x)
'''
np.set_printoptions(precision=3)
print('start')
a1, a2 = task_c(P,x,10)
print(a1, '\n', a2)
a1, a2 = task_c(P,x,40)
print('\n\n',a1, '\n', a2)
'''
def inf_gain_at_k(P, x, k):
    FIX = 100000
    y1, y2 = task_c(P, x, k)
    #print(y1, y2)
    marg_1, marg_2 = task_a(P, x)
    s1 = 0
    s2 = 0
    for n in range(0, 50):
        s1 += 5000*y1[n]
        s2 += 5000*y2[n]

    #print("s1: {} -> {}, s2: {} -> {}".format(s1, min([FIX, s1]), s2, min([FIX, s2])))
    
    total = min([FIX, s1])*marginal_prob(k) + min([FIX, s2])*(1-marginal_prob(k))
    #print(total)

    return total


def task_e(P, x):
    totals = []
    for k in range(50):
        totals.append(inf_gain_at_k(P, x, k))

    print(totals)
    print(len(totals))
    plt.figure()
    plt.plot(range(1,51), totals)
    plt.show()

task_e(P, x)