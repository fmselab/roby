'''
Created on Jan 24, 2021

@author: AngeloGargantini
'''
import math
from matplotlib.pyplot import plot
import numpy as np
#
def const1(x):

    return 1

l = []

def eval(function,lb,ub, threshold, minstep):
    #
    if (ub-lb <minstep):
        return
    #
    print("eval function in " +str(lb) + " " +str(ub))
    p1 = function(lb)
    p2 = function(ub)
    ## add these to list 
    if lb not in l:
        l.append(lb)
    if ub not in l:
        l.append(ub)
    # compute distance among p1 and p2
    a = math.sqrt( ((p1 -p2)**2)+((ub-lb)**2) )
    b = a/3
    # compute distance of the
    alpha = math.atan((p1 -p2)/(ub-lb))
    distfromcenter = math.sqrt( a**2* math.sin(alpha)**2+b**2* math.cos(alpha)**2)
    # distance from the threshold
    Ofromth = math.fabs((p1+p2)/2 - threshold)
    print(" in " +str(lb) + " " +str(ub))
    if (Ofromth > distfromcenter):
        return
    else:
        eval(function,lb,(ub+lb)/2, threshold, minstep)
        eval(function,(ub+lb)/2, ub, threshold, minstep)


if __name__ == '__main__':
    threshold = 0.5
#    eval(lambda a : 1 , 0, 1, 0.5,0.1)
#    eval(lambda a : 1 - a/2, 0, 1, 0.6,0.05)
    f = lambda a : (1 - a/2 + 0.1*math.sin(a*50))
    #f = lambda a : (1 + (- a/2 if a<0.5  else + (-0.5 + a/2) ) + 0.05*math.sin(a*50))
    eval(f, 0, 1, threshold,0.05)
    import matplotlib.pyplot as plt
    t = np.arange(0., 1., 0.01)
    plt.plot(t, np.vectorize(f)(t))
    plt.plot(t, np.vectorize(lambda a: threshold)(t))
    l.sort()
    plt.plot(l,np.vectorize(f)(l),'bs')
    plt.show()
    pass 