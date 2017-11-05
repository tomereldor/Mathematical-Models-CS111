##Tomer Eldor #### Disease Outbreak####

#IMPORTS
import random
import matplotlib.pyplot as plt

###set initial conditions###
S = 1000 #number of susceptibles
I = 5 #starting infectious people
R = 0 #starting recovered
t = 0 #initial time
#population size:
N = float(S+I+R)

###Disease transmission parameters###
#infection rate
r = 0.5
#recovery rate
g = 0.2

#Initiating history lists of each parameter:
s_hist = [] 
i_hist = [] 
r_hist = [] 
I_added = [] #number of new Infected at step t

#### BRINGING IN SOME STOCHASTIC / RANDOMNESS #####
#loop until no more infected (if equilibrium, python wil break for us)
while I > 0:
    i_new = 0
    #randomly drawing transitions from S to I
    for i in range(S):
        if random.random() < r*(I/N): #
            i_new += 1
    #randomly drawing transitioning from I to R:
    r_new = 0
    for i in range(I):
        if random.random() < g:
            r_new += 1

    #now, updating the quantities with the new transitioned individuals, and add it to its list:
    S = S-i_new
    s_hist.append(S)

    I = I+(i_new - r_new)
    i_hist.append(I)

    R = R+r_new
    r_hist.append(R)

    I_added.append(i_new)

#PLOTTING#
plt.plot(i_hist)
plt.plot(s_hist)
plt.plot(r_hist)
plt.show()
plt.plot()
