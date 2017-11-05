
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing data into panda array
data = pd.read_csv("/Users/tomereldor/Google Drive/Academics (Tomer)/0-2017 Buenos Aires Sem2.2/CS111 Math/cs111-svm-dataset.csv")

#plotting the first classification - how amazing, it can c
old = data.plot(x='X1', y='X2', kind="scatter", c='Classification', cmap='Spectral', colorbar=False, title='Classification of Given Data')
#plt.show()

## Partial Derivatives of m1, m2, b

#D of m1
def d_m1(x1, x2, y, m1, m2, b):
    A = np.exp(-y * (m1*x1 + m2*x2 + b))
    return np.mean(-y * x1 * A * (1 / (1 + A)))

# D of m2
def d_m2(x1, x2, y, m1, m2, b):
    A = np.exp(-y * (m1 * x1 + m2 * x2 + b))
    return np.mean(-y * x2 * A * (1 / (1 + A)))

# D of b
def d_b(x1, x2, y, m1, m2, b):
    A = np.exp(-y * (m1 * x1 + m2 * x2 + b))
    return np.mean(-y * A * (1 / (1 + A)))

# extracting variables from origianl data
x1 = data['X1'].values
x2 = data['X2'].values
y = data['Classification'].values
y[y == 0] = -1

### Initializing Variables

#initializing null / initial values for
precision = 0.00001
alpha = 0.1 #initializing step_size as alpha; we want to set pretty small to not "skip" the optimal minimum
#first values to start with
m1, m2, b = 0.5, 0.5, 1
m1_p, m2_p, b_p = 0, 0, 0
hist_array = np.array([]) #initializing a history array of arrived values

#performing the descent, until we reach desired percision
#(checking precision by checking if the difference between this iteration values to the previous interation's values are as small as desired precision

steps = 0
#global m1, m2, b, m1_p, m2_p, b_p,alpha,precision,hist_array

#initializing null / initial values for
precision = 0.00001
alpha = 0.1 #initializing step_size as alpha; we want to set pretty small to not "skip" the optimal minimum
#first values to start with
m1, m2, b = 0.5, 0.5, 1
m1_p, m2_p, b_p = 0, 0, 0
hist_array = [] #initializing a history array of arrived values

#print (m1, m2, b, m1_p, m2_p, b_p,alpha,precision,hist_array)

while  (abs(b - b_p) > precision)  or (abs(m1 - m1_p) > precision) or (abs(m2 - m2_p) > precision) :
    #initializing with previous values
    m1_p, m2_p, b_p = m1, m2, b
    #new values, according to step size times partial
    m1_t = m1 - alpha * d_m1(x1, x2, y, m1, m2, b)
    m2_t = m2 - alpha * d_m2(x1, x2, y, m1, m2, b)
    b_t = b - alpha * d_b(x1, x2, y, m1, m2, b)
    m1, m2, b = m1_t, m2_t, b_t     #setting into normal values
    steps += 1      #counter
    #saving these to history array
    fi = (m1*x1) + (m2*x2) +b
    #print fi
    hist_array.append([steps,fi])
print m1
print
print fi


plt.plot(hist_array)
plt.scatter(hist_array[:,0],hist_array[:,1], c=y, cmap='Spectral')
plt.title("History of F")
plt.xlabel("i")
plt.ylabel("F")
plt.show()


def findx2(x1, m1, m2, b):
    # 1. we assume that f(x1, x2) = 0
    # 2. we turn the line equation to find the x2 value for a given x1, m1, m2, b
    return (m1 * x1 + b) / -m2

#Preparing to plot: finding the line on the hyperplane of x1
hyp_x1 = np.linspace(-35, 35, 20)
line = findx2(hyp_x1, m1, m2, b)

#Plotting the graph
plt.plot(hyp_x1, line)
plt.scatter(x1, x2, c=y, cmap='Spectral')
plt.title("Support Vector Machine")
plt.xlabel("X1")
plt.ylabel("X2")
plt.show()
