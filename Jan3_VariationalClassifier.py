# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 14:12:48 2020

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

np.random.seed(42)

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

def R(y, probs, b, R):
    p = 1 - probs
    y = 2*y - 1
    loss = 0
    for k in range(len(y)):
        if y[k] == 1:
            x = (math.sqrt(R)*(.5 - (probs[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        else:
            x = (math.sqrt(R)*(.5 - (p[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        loss = loss + (1 / (1 + math.exp(-x)))
    loss = loss / len(probs)
    return loss

def J(theta, X, Y, qubits, nr_layers, batch_i, shots, key):
    simulator = cirq.Simulator()
    p = np.zeros([len(batch_i),])
    for i in range(len(batch_i)):
        p_hold =0
        c = cirq.Circuit()
        c.append(circuit(qubits, X[batch_i[i]], theta, nr_layers))
        results = simulator.run(c, repetitions=shots)
        counter = (results.multi_measurement_histogram(keys="012"))
        for j in counter:
            if j.count(1) % 2 == 1:
                p_hold += counter[j]
        p[i] = p_hold/shots
    loss = squared_loss(Y[batch_i], p)
    return loss

def J_x(theta, b, X, Y, qubits, nr_layers, batch_i, shots, key):
    simulator = cirq.Simulator()
    p = np.zeros([len(batch_i),])
    for i in range(len(batch_i)):
        p_hold =0
        c = cirq.Circuit()
        c.append(circuit(qubits, X[batch_i[i]], theta, nr_layers))
        results = simulator.run(c, repetitions=shots)
        counter = (results.multi_measurement_histogram(keys="012"))
        for j in counter:
            if j.count(1) % 2 == 1:
                p_hold += counter[j]
        p[i] = p_hold/shots
    loss = R(Y[batch_i], p, b, shots)
    return loss

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
        
nr_qubits = 3
nr_layers = 2
batch_size = 10
shots = 100
iterations = 20

key = ""
for i in range(nr_qubits):
    key += str(i)

#Set up input and output qubits.
qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]

df = pd.read_csv("QA_data.csv")
X = df.iloc[:,:3].to_numpy()
Y = df.iloc[:,3].to_numpy()

m = int(np.round((2/3)*len(X)))
train = np.random.randint(0, len(X), (m,))
X_t = X[train,:]
Y_t = Y[train]

nr_par = (nr_qubits*2)*(nr_layers+1)
init_theta = np.random.rand(nr_par,)*(2*pi)
theta = init_theta
b = 0
eye = np.eye(nr_par)
c = 0.1
a = 2*pi*0.1
alpha = 0.602
gamma = 0.101
iz = np.array(range(m))
plot_ix = 10
P = int(iterations/plot_ix)
tot_loss = np.zeros(P)
Tot_Loss = np.zeros(iterations)


iw = 0
#start = time.time()
for k in range(1,iterations+1):
    start = time.time()   
    printProgressBar((k-1)*nr_par, iterations*nr_par, prefix = 'Progress:', suffix = 'Complete', length = 50)
    batch_ix = np.random.randint(0, len(X_t), (batch_size,))
    #batch_ix = iz
    c_n = c/(k**(gamma))
    a_n = a/(k**(alpha))
    z_n = c/(k**(alpha))
    gradient = np.zeros(nr_par)
    
    for i in range(nr_par):
        printProgressBar((k-1)*nr_par+i, iterations*nr_par, prefix = 'Progress:', suffix = 'Complete', length = 50)
        loss_plus = J_x(theta+c_n*eye[:,i], b, X_t, Y_t, qubits, nr_layers, batch_ix, shots, key)
        loss_minus= J_x(theta-c_n*eye[:,i], b, X_t, Y_t, qubits, nr_layers, batch_ix, shots, key)
        gradient[i] = (loss_plus - loss_minus)/(2*c_n)
        
    
    loss_plus = J_x(theta, b+c_n, X_t, Y_t, qubits, nr_layers, batch_ix, shots, key)
    loss_minus= J_x(theta, b-c_n, X_t, Y_t, qubits, nr_layers, batch_ix, shots, key)
    grad_b = (loss_plus - loss_minus)/(2*c_n) 

    theta = (theta - a_n*gradient) % (2*pi)
    b = b - z_n*grad_b 
    
    #Tot_Loss[k-1] = J_x(theta, b, X_t, Y_t, qubits, nr_layers, iz, shots, key)
    if k % plot_ix == 0:
        tot_loss[iw] = J_x(theta, b, X_t, Y_t, qubits, nr_layers, iz, shots, key)
        iw=iw+1
    end = time.time()
    #print("iteration: ",k+1," time: ", np.round(end - start))
    
printProgressBar(iterations*nr_par, iterations*nr_par, prefix = 'Progress:', suffix = 'Complete', length = 50)

print(end - start)
#print("init_theta - end_theta: ")
#print(np.around(init_theta-theta,3))

fig = plt.figure(figsize=(15,10))
ax = plt.gca()
plt.plot(range(1,P+1), tot_loss, 'g-', markersize=2)