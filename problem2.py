import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.style.use('ggplot') #to be used for pthe plot in task a

P = np.matrix([[0.95, 0.05], [0.0, 1.0]])

def P_x(k):
    """Returns the probability of X_k = 1
    """
    if (k == 1):
        return 0.99
    else:
        return 0.99 * 0.95**(k-1)

def simulate_hill(n_sim):
    """
    Simulate 25 realizations of the Markov chain. 
    """
    #list to store each simulation
    realizations = []

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

def P_bw(k):
    """Returns the backward transition matrix at step k
    """
    PB = np.matrix([[1.0,0.0],[0.0,0.0]])
    PB[1,0]=P[0,1]*P_x(k-1)/(1-P_x(k))
    PB[1,1]=P[1,1]*(1-P_x(k-1))/(1-P_x(k))
    return PB

def sensor_prob(k):
    """Compute forward and backward probability at each point, given that we
    have observed state X_k
    """
    v1 = np.matrix([1,0]) # X_k = 1
    v2 = np.matrix([0,1]) # X_k = 2

    # Prepare result vectors 
    y1 = np.zeros((50,))
    y2 = np.zeros((50,))
    
    lim = 50
    
    plac = k
    while(plac > 1):
        v1 = v1 * P_bw(plac)
        v2 = v2 * P_bw(plac)
        y1[plac-2] = v1[0, 1]
        y2[plac-2] = v2[0, 1]
        plac -= 1
    plac = k-1
    
    # Reset state vectors
    v1 = np.matrix([1,0])
    v2 = np.matrix([0,1])
    
    while(plac < lim):
        y1[plac] = v1[0, 1]
        y2[plac] = v2[0, 1]
        v1 = v1 * P
        v2 = v2 * P
        plac += 1

    return y1, y2


def information_gain(k):
    """As defined by V_k in the tasks
    """
    y1, y2 = sensor_prob(k)
    
    s1 = 5000*sum(y1)
    s2 = 5000*sum(y2)
        
    total = min([100000, s1])*P_x(k) + min([100000, s2])*(1-P_x(k))

    return total


def plot_c(k):
    y1, y2 = sensor_prob(k)
    print('Length og y1 and y2: ', len(y1), ', ', len(y2))
    plt.style.use("ggplot")
    plt.title('Forward and backward propabilities as a function of $l$')
    plt.ylabel("$P(X_l = 2)$")
    plt.xlabel("$l$")
    plt.plot(range(1,51), y1)
    plt.plot(range(1,51), y2)
    plt.legend(["$P(X_l = 2 | X_{20} = 1)$", "$P(X_l = 2 | X_{20} = 2)$"])
    plt.show()


def task_a():
    plt.figure()
    plt.title('Marginal probabilities for $P(X_n = 1)$ as function of $n$.', fontsize=14)
    plt.xlabel('$n$')
    plt.ylabel('$P(X_n = 1)$')
    plt.plot(range(1,51), [P_x(k) for k in range(1,51)])


def task_b():
    '''
    plotting the realizations from task 2b with imshow
    '''
    image, means = simulate_hill(25)

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
    

def task_c():
    k = 20
    y1, y2 = sensor_prob(k)
    print('Length og y1 and y2: ', len(y1), ', ', len(y2))
    plt.style.use("ggplot")
    plt.title('Forward and backward propabilities as a function of $l$')
    plt.ylabel("$P(X_l = 2)$")
    plt.xlabel("$l$")
    plt.plot(range(1,51), y1)
    plt.plot(range(1,51), y2)
    plt.legend(["$P(X_l = 2 | X_{20} = 1)$", "$P(X_l = 2 | X_{20} = 2)$"])
    plt.show()

    

def task_d():
    '''
    This function gets the probabilities from task a 
    and sums over these with a cost of 5000 per road.
    '''
    cost = min([1000000, 5000*sum([P_x(k) for k in range(1,51)])])
    
    print('\ntotal cost = ', cost)
    print('optimal choice is: ')
    if cost < 100000:
        print('5000 per road ')
    else:
        print('fixed price of 100000')


def task_e():
    plt.figure()
    plt.plot(range(1,51), [information_gain(k) for k in range(1,51)])
    plt.title("$V_k = \sum_{i=1}^{2} min \{1000000, 5000\sum_{n=1}^{50}P(X_n = 2 | X_k = i)\}P(X_k=i)$")
    plt.xlabel("$k$")
    plt.ylabel("$V_k$")
    plt.show()

    
if __name__ == "__main__":
    task_a()
    task_b()
    task_c()
    task_d()
    task_e()
