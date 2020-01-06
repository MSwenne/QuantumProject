# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 13:49:33 2020

@author: vshas
"""

import tensorflow as tf
import cirq
import numpy as np
import random
import sympy
from math import pi
import math
import time
import pandas as pd
import matplotlib.pyplot as plt
#matplotlib inline

#np.random.seed(111) # 42d

def U_phi(q, W):
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent=W[i]/pi)
        yield rot(q[i])
    for i in range(len(q)-1):
        for j in range(i+1,len(q)):
            rot = cirq.ZPowGate(exponent=((pi-W[i])*(pi-W[j]))/pi)
            yield rot.on(q[j]).controlled_by(q[i])
            
def fancy_U(q, W):
    for i in range(len(q)):
        yield cirq.H(q[i])
    yield U_phi(q, W)
    for i in range(len(q)):
        yield cirq.H(q[i])
    yield U_phi(q, W)

def W_theta(q, theta):
    for i in range(len(q)):
        yield cirq.CZ.on(q[(i+1)%len(q)],q[i])
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent=theta[2*i]/pi)
        yield rot(q[i])
        rot = cirq.Ry(theta[2*i+1])
        yield rot(q[i])

def measure(q):
    for i in range(len(q)):
        yield cirq.measure(q[i], key=str(i))
        
def circuit(q, W, theta, layers):
    yield fancy_U(q,W)
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent = theta[2*i]/pi)
        yield rot(q[i])
        rot = cirq.Ry(theta[2*i + 1])
        yield rot(q[i])
    
    for i in range(1,layers+1):
        yield W_theta(q, theta[range(6*(i),6*(i+1))])
    yield measure(q)

def abs_loss(labels, predictions):
    loss=0
    pred = np.round(predictions)
    for l, p in zip(labels, pred):
        loss = loss + np.abs(l - p)
    loss = loss / len(labels)
    return loss

def squared_loss(labels, predictions):
    loss =0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p)**2
    loss = loss/ len(labels)
    return loss

def assign_label(p, Y, b):
    Y_pm = 2*Y -1
    labels = np.ones([len(p),])*-1
    for i in range(len(p)):
        if (p[i] > ((1 - p[i]) - Y_pm[i]*b)):
            labels[i] = 1
    return labels    

def R(y, probs, b):
    p = 1 - probs
    y = 2*y - 1
    loss = 0
    #R = shots
    for k in range(len(y)):
        if y[k] == 1:
            x = (math.sqrt(R)*(.5 - (probs[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        else:
            x = (math.sqrt(R)*(.5 - (p[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        loss = loss + (1 / (1 + math.exp(-x)))
    loss = loss / len(probs)
    return loss

def J_w(theta, X, qubits, nr_layers, shots):
    simulator = cirq.Simulator()
    p = np.zeros([len(X),])
    for i in range(len(X)):
        p_hold =0
        c = cirq.Circuit()
        c.append(circuit(qubits, X[i], theta, nr_layers))
        results = simulator.run(c, repetitions=shots)
        counter = (results.multi_measurement_histogram(keys="012"))
        for j in counter:
            if j.count(1) % 2 == 1:
                p_hold += counter[j]
        p[i] = p_hold/shots
    return p


def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
	"""
	Call in a loop to create terminal progress bar
	@params:
		iteration   - Required  : current iteration (Int)
		total	   - Required  : total iterations (Int)
		prefix	  - Optional  : prefix string (Str)
		suffix	  - Optional  : suffix string (Str)
		decimals	- Optional  : positive number of decimals in percent complete (Int)
		length	  - Optional  : character length of bar (Int)
		fill		- Optional  : bar fill character (Str)
	"""
	percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
	filledLength = int(length * iteration // total)
	bar = fill * filledLength + '-' * (length - filledLength)
	print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
	# Print New Line on Complete
	if iteration == total: 
		print()
  
np.random.seed(239)      
nr_qubits = 3
nr_layers = 4
batch_size = 100
shots = 2000
iterations = 10

key = ""
for i in range(nr_qubits):
    key += str(i)

#Set up input and output qubits.
qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]

df = pd.read_csv("QA_data_x.csv")
X = df.iloc[:,:3].to_numpy()
Y = df.iloc[:,3].to_numpy()

indexes = np.array(range(len(X)))
m = int(np.round((4/5)*len(X)))
#train = np.random.randint(0, len(X), (m,))
train = np.random.choice(len(X), m, replace=False)
test = indexes[~np.isin(indexes,train)]
X_t = X[train,:]
Y_t = Y[train]

nr_par = (nr_qubits*2)*(nr_layers+1)
theta_star = np.load("theta_star.npy")
init_theta = np.random.rand(nr_par,)*(2*pi)
i#nit_theta = np.ones([nr_par,])*pi
theta = init_theta
b = 0
theta = np.append(theta,b)
eye = np.eye(nr_par)
c = 0.1
a = 2*pi*0.1
alpha = 0.602
gamma = 0.101
parameters = np.array([a,c,alpha,gamma])
iz = np.array(range(m))
plot_ix = 5
P = int(iterations/plot_ix)
tot_loss = np.zeros(P)
Tot_Loss = np.zeros(iterations)


####################################################
#   Calibration Step
####################################################

#stat = 25
#hold_c0 = parameters[0]
#initial_c = parameters[1]
#delta_obj = 0
#for i in range(stat):
#    print(i)
#    delta = 2 * np.random.randint(2, size = len(theta)) - 1
#    obj_plus = J_w(theta+initial_c*delta, X_t, qubits, nr_layers, shots)
#    obj_minus = J_w(theta+initial_c*delta, X_t, qubits, nr_layers, shots)
#    loss_p = squared_loss(Y_t, obj_plus)
#    loss_m = squared_loss(Y_t, obj_minus)
#    delta_obj += np.absolute(loss_p - loss_m) / stat
    
#c_new = hold_c0 * 2 / delta_obj * initial_c    
    
####################################################

a = 2.5
z = a/pi
loss_est = 0
iw = 0
start = time.time()
printProgressBar(1, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
for k in range(1,iterations+1): 
    printProgressBar((k-1)*nr_par, iterations*nr_par, prefix = 'Progress:', suffix = 'Complete', length = 50)
    #batch_ix = np.random.randint(0, len(X_t), (batch_size,))
    batch_ix = iz
    
    c_n = c/(k**(gamma))
    a_n = a/(k**(alpha))
    z_n = z/(k**(alpha))
    gradient = np.zeros(nr_par)
    delta_n = 2*np.random.randint(2, size = nr_par+1) - 1
    
    p_plus = J_w(theta+c_n*delta_n, X_t, qubits, nr_layers, shots)
    p_minus= J_w(theta-c_n*delta_n, X_t, qubits, nr_layers, shots)
    loss_plus = R(Y_t, p_plus, theta[-1]) #squared_loss(Y_t, p_plus) #R(Y_t, p_plus, theta[-1])
    loss_minus= R(Y_t, p_minus, theta[-1])#squared_loss(Y_t, p_minus)#R(Y_t, p_minus, theta[-1])

    grad = ((loss_plus - loss_minus)/(2*c_n))/delta_n
    theta[1:-1] = (theta[1:-1] - a_n*grad[1:-1]) #% (2*pi)
    theta[-1] = (theta[-1] - z_n*grad[-1]) 
    # parameter b is probably taking to large of a steps.
   
    Tot_Loss[k-1] = (loss_plus + loss_minus)/2
    printProgressBar(1+k, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

end = time.time()    
printProgressBar(iterations, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

print(end - start)

Y_pm = 2*Y_t - 1
p_hat = J_w(theta, X_t, qubits, nr_layers, 1000)

fig = plt.figure(figsize=(15,10))
ax = plt.gca()
plt.plot(range(1,iterations+1), Tot_Loss, 'g-', markersize=2)