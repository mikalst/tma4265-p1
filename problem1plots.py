# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

titlesize = 18
textsize = 12.5

def plot1(xvalues, yvalues, title, xlabel, ylabel, opt, xl,
          xu, yl, yu):
    plt.figure()
    plt.style.use("ggplot")
    plt.grid(True)
    plt.title(title, fontsize = titlesize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    
    xopt = round(opt)
    yopt = yvalues[xopt - 1]

    plt.annotate("Optimal choise,\nk = {},\np = {}".format(xopt, round(yopt, 4)), xy = (xopt, yopt),
                 xytext = (xopt + 0.5, yopt + 0.05), fontsize = 12.5, color = "red",
                 arrowprops = dict(facecolor = "black", shrink = 0.05))
    
    plt.plot(xvalues, yvalues, "ro", color = "red")
    plt.plot(xvalues, yvalues, "b", color = "red")
    plt.show()

def plot2(xvalues1, yvalues1, xvalues2, yvalues2, title,
          xlabel, ylabel, opt1, opt2, xl, xu, yl, yu):
    plt.figure()
    plt.style.use("ggplot")
    plt.title(title, fontsize = titlesize)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim(xl, xu)
    plt.ylim(yl, yu)
    
    xopt1 = round(opt1)
    xopt2 = round(opt2)
    yopt1 = yvalues1[xopt1 - 1]
    yopt2 = yvalues2[xopt2 - 1]
    plt.annotate("$N = 30$", xy = (34, 0.45), fontsize = 17, color = "red")
    plt.annotate("$N = 40$", xy = (34, 0.40), fontsize = 17, color = "blue")
    plt.annotate("Optimal choise,\nk = {},\np = {}".format(xopt1, round(yopt1, 4)), xy = (xopt1, yopt1),
                                             xytext = (xopt1 - 8.5, yopt1 + 0.04), fontsize = textsize, color = "red",
                                             arrowprops = dict(facecolor = "black", shrink = 0.05) )
    plt.annotate("Optimal choise,\nk = {},\np = {}".format(xopt2, round(yopt2, 4)), xy = (xopt2, yopt2),
                                             xytext = (xopt2, yopt1 + 0.04), fontsize = textsize, color = "blue",
                                             arrowprops = dict(facecolor = "black", shrink = 0.05) )
    
    
    plt.plot(xvalues1, yvalues1, "ro", color = "red")
    plt.plot(xvalues1, yvalues1, "b", color = "red")
    plt.plot(xvalues2, yvalues2, "ro", color = "blue")
    plt.plot(xvalues2, yvalues2, "b", color = "blue")
    plt.show()










