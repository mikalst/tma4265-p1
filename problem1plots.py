# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def plot(xvalues, yvalues, title, xlabel, ylabel, opt, xl, xu, yl, yu):
    
    fig = plt.figure()
    plt.style.use("ggplot")
    plt.grid(True)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    
    xopt = round(opt)
    yopt = yvalues[xopt - 1]
    
    plt.annotate("optimal choise,\n({}, {})".format(xopt, round(yopt, 4)), xy = (xopt, yopt),
                 xytext = (xopt + 1, yopt + 0.05), fontsize = 20,
                 arrowprops = dict(facecolor = "black", shrink = 0.05))
    
    plt.plot(xvalues, yvalues)
    
    plt.show()

