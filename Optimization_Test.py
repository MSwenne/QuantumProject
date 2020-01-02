# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 20:22:22 2020

@author: vshas
"""


import tensorflow as tf
import cirq
import numpy as np
import random
import sympy
from math import pi
import pandas as pd
import time 
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

def circuit_x(q, W, theta, layers):
    #m = len(q)
    yield fancy_U(q,W)
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent = theta[2*i]/pi)
        yield rot(q[i])
        rot = cirq.Ry(theta[2*i + 1])
        yield rot(q[i])
    
    for j in range(1,layers+1):
        yield W_theta(q, theta[range(6*(j),6*(j+1))])
    yield measure(q)


def circuit(q, W, theta):
    yield fancy_U(q, W)
    for i in range(len(theta)):
        if i == 0:
            for j in range(len(q)):
                rot = cirq.ZPowGate(exponent=theta[i][j][0])
                yield rot(q[j])
                rot = cirq.Ry(theta[i][j][1])
                yield rot(q[j])
        else:
            yield W_theta(q, theta[i])
    yield measure(q)
    
def squared_loss(labels, predictions):
    loss =0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p)**2
    loss = loss/ len(labels)
    return loss

def J(theta, X, Y, nr_qubits, nr_layers, batch_ix, shots):
    simulator = cirq.Simulator()
    for i in range(len(batch_ix)):
        p_hold =0
        c = cirq.Circuit()
        c.append(circuit_x(qubits, X[batch_ix[i]], theta, nr_layers))
        for k in range(shots):
            results = simulator.run(c)
            for j in range(nr_qubits):
                count = results.histogram(key=str(j))[0]
            if count % 2 != 0:
                p_hold = p_hold +1
        p[i] = p_hold/shots
    loss = squared_loss(Y[batch_index], p)
    return loss

## From here No def's but implementation.
    
nr_qubits = 3
nr_layers = 2
batch_size = 15
repetitions = 100
shots = 50

#Set up input and output qubits.
qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]

df = pd.read_csv("QA_data.csv")
X = df.iloc[:,:3].to_numpy()
Y = df.iloc[:,3].to_numpy()

nr_par = (nr_qubits*2)*(nr_layers+1)
theta = np.random.rand(nr_par,)*(2*pi)
p = np.zeros([batch_size,])
eye = np.eye(nr_par)
c = 0.1
a = 2*pi*0.1
alpha = 0.602
gamma = 0.101

for k in range(1,3):
    print("iteration:",k)
    batch_ix = np.random.randint(0, len(X), (batch_size,))
    c_n = c/(k**(0.602))
    a_n = a/(k**(0.101))
    
    gradient = np.zeros([nr_par,])
    for i in range(nr_par):
        start = time.time()
        J_plus = J(theta+c_n*eye[:,i], X, Y, nr_qubits, nr_layers, batch_ix, shots)
        J_minus= J(theta-c_n*eye[:,i], X, Y, nr_qubits, nr_layers, batch_ix, shots)
        gradient[i] = (J_plus - J_minus)/(2*c_n)
        end = time.time()
        print("gradient component:",i," loss: ",J_plus," time: ",end-start)
        

    theta = theta - a_n*gradient








