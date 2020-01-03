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

np.random.seed(32)


def measure(q):
    for i in range(len(q)):
        yield cirq.measure(q[i], key=str(i))

def U_phi_it(q, W):
    rot1 = cirq.ZPowGate(exponent=W[0]/pi)
    rot2 = cirq.ZPowGate(exponent=W[1]/pi)
    rot3 = cirq.ZPowGate(exponent=W[2]/pi)
    rot12= cirq.ZPowGate(exponent=((pi-W[0])*(pi-W[1]))/pi)
    rot13= cirq.ZPowGate(exponent=((pi-W[0])*(pi-W[2]))/pi)
    rot23= cirq.ZPowGate(exponent=((pi-W[1])*(pi-W[2]))/pi)
    # apply
    yield cirq.H(q[0])
    yield cirq.H(q[1])
    yield cirq.H(q[2])
    yield rot1(q[0])
    yield rot2(q[1])
    yield rot3(q[2])
    yield rot12.on(q[0]).controlled_by(q[1])
    yield rot13.on(q[0]).controlled_by(q[2])
    yield rot23.on(q[1]).controlled_by(q[2])
    yield cirq.H(q[0])
    yield cirq.H(q[1])
    yield cirq.H(q[2])
    yield rot1(q[0])
    yield rot2(q[1])
    yield rot3(q[2])
    yield rot12.on(q[0]).controlled_by(q[1])
    yield rot13.on(q[0]).controlled_by(q[2])
    yield rot23.on(q[1]).controlled_by(q[2])

def W_theta_it(q, theta):
    m = len(q)
    for i in range(m):
        rot1 = cirq.ZPowGate(exponent=theta[2*i]/pi)
        yield rot1(q[i])
        rot2 = cirq.Ry(theta[2*i+1])
        yield rot2(q[i])
    
    yield cirq.CZ.on(q[0],q[1])
    yield cirq.CZ.on(q[0],q[2])
    yield cirq.CZ.on(q[1],q[2])
    
    for i in range(m):
        rot1 = cirq.ZPowGate(exponent=theta[6+(2*i)]/pi)
        yield rot1(q[i])
        rot2 = cirq.Ry(theta[6+(2*i)+1])
        yield rot2(q[i])
        
    yield cirq.CZ.on(q[0],q[1])
    yield cirq.CZ.on(q[0],q[2])
    yield cirq.CZ.on(q[1],q[2])
    
    for i in range(m):
        rot1 = cirq.ZPowGate(exponent=theta[12+(2*i)]/pi)
        yield rot1(q[i])
        rot2 = cirq.Ry(theta[12+(2*i)+1])
        yield rot2(q[i])

def circuit_it(q, W, theta):
    yield U_phi_it(q, W)
    yield W_theta_it(q, theta)
    yield measure(q)

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
    loss = squared_loss(Y[batch_ix], p)
    return loss

## From here No def's but implementation.
    
nr_qubits = 3
nr_layers = 2
batch_size = 1
repetitions = 100
shots = 30

#Set up input and output qubits.
qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]

df = pd.read_csv("QA_data.csv")
X = df.iloc[:,:3].to_numpy()
Y = df.iloc[:,3].to_numpy()

nr_par = (nr_qubits*2)*(nr_layers+1)
#theta = np.random.rand(nr_par,)*(2*pi)
theta_0 = np.ones([nr_par,])*pi
theta = theta_0
p = np.zeros([batch_size,])
eye = np.eye(nr_par)
c = 0.1
a = 2*pi*0.1
alpha = 0.602 #0.3
gamma = 0.101
nr_epochs = 100

np.zeros([nr_epochs,])

#batch_ix = np.random.randint(0, len(X), (batch_size,))
for k in range(1,nr_epochs+1):
    #print("iteration:",k)
    batch_ix = np.random.randint(0, len(X), (batch_size,))
    c_n = c/(k**(0.602))
    a_n = a/(k**(0.101))
    start = time.time()
    gradient = np.zeros([nr_par,])
    for i in range(nr_par):
        print(".", end='')
        theta_plus = np.minimum(theta+c_n*eye[:,i],2*pi)
        theta_minus= np.maximum(theta-c_n*eye[:,i],-2*pi)
        J_plus = J(theta_plus, X, Y, nr_qubits, nr_layers, batch_ix, shots)
        J_minus= J(theta_minus, X, Y, nr_qubits, nr_layers, batch_ix, shots)
        gradient[i] = (J_plus - J_minus)/(2*c_n)
    print(".")
    end = time.time()
    print("iteration:",k," loss: ",J_plus," time: ",end-start)
    theta = np.minimum(np.maximum(theta - a_n*gradient, -2*pi),2*pi)








