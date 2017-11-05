from discreteMarkovChain import markovChain
import numpy as np


d_matrix = np.genfromtxt('/Users/tomereldor/Downloads/Cgraph.csv', delimiter=',')
d_matrix[0,0] = 0


P = np.array([[0.5,0.5],[0.6,0.4]])
mc = markovChain(d_matrix)
mc.computePi('linear') #We can also use 'power', 'krylov' or 'eigen'
print(mc.pi)


from discreteMarkovChain import partition

class randomWalkNumpy(markovChain):
    #Now we do the same thing with a transition function that returns a 2d numpy array.
    #We also specify the statespace function so we can use the direct method.
    #This one is defined immediately for general n.
    def __init__(self,m,M,n,direct=True):
        super(randomWalkNumpy, self).__init__(direct=direct)
        self.initialState = m*np.ones(n,dtype=int)
        self.n = n
        self.m = m
        self.M = M
        self.uprate = 1.0
        self.downrate = 1.0

        #It is useful to define the variable 'events' for the the transition function.
        #The possible events are 'move up' or 'move down' in one of the random walks.
        #The rates of these events are given in 'eventRates'.
        self.events = np.vstack((np.eye(n,dtype=int),-np.eye(n,dtype=int)))
        self.eventRates = np.array([self.uprate]*n+[self.downrate]*n)

    def transition(self,state):
        #First check for the current state which of the 'move up' and 'move down' events are possible.
        up = state < self.M
        down = state > self.m
        possibleEvents = np.concatenate((up,down))  #Combine into one boolean array.

        #The possible states after the transition follow by adding the possible 'move up'/'move down' events to the current state.
        newstates = state+self.events[possibleEvents]
        rates = self.eventRates[possibleEvents]
        return newstates,rates

    def statespace(self):
          #Each random walk can be in a state between m and M.
          #The function partition() gives all partitions of integers between min_range and max_range.
          min_range = [self.m]*self.n
          max_range = [self.M]*self.n
          return partition(min_range,max_range)


mc = randomWalkNumpy(0,2,n=2)
mc.computePi('linear')
mc.printPi()
