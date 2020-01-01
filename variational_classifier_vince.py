# -*- coding: utf-8 -*-
"""
Created on Tue Dec 31 11:40:05 2019

@author: vshas
"""

import cirq
import numpy as np
import pandas as pd
import random
import sympy
from math import pi

# incorporating the data

df = pd.read_csv('QA_data.csv')
X = df.iloc[:,0:3].to_numpy()
y = df.iloc[:,3].to_numpy()



np.random.seed(42)

nr_qubits = 3
num_layers = 1
batch_size = 5
data_point = X[3,:]
par = np.zeros([18,2])

## abstractions of circuit parts

# inladen van de data
def U_phi(q, W): 
    print(W)
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent=(W[i]/pi))
        yield rot(q[i])
    for i in range(len(q)):
        if i < len(q)-1:
            rot = cirq.ZPowGate(exponent=((pi-W[i])*(pi-W[i+1])/pi))
            yield rot.on(q[i+1]).controlled_by(q[i])
        else:
            rot = cirq.ZPowGate(exponent=((pi-W[i])*(pi-W[0])/pi))
            yield rot.on(q[i]).controlled_by(q[0])

def U_data(q, W):
    for i in range(len(q)):
        yield cirq.H(q[i])
    yield U_phi(q, W)
    for i in range(len(q)):
        yield cirq.H(q[i])
    yield U_phi(q, W)

# variational circuit

def U_ent(q):
    for i in range(len(q)):
        if i < len(q)-1:
            yield cirq.Z(q[i+1]).controlled_by(q[i])
        else:
            yield cirq.Z(q[i]).controlled_by(q[0])

def U_theta_mt(q_0, par):
    rot_z = cirq.ZPowGate(exponent=(par[0]/pi))
    rot_y = cirq.Rz(par[1]) 
    yield rot_z(q_0)
    yield rot_y(q_0)

def U_loc(q, par, k):
    m = len(q)*k
    for i in range(len(q)):
        yield U_theta_mt(q[i], par[m+i,:])
    

def W_theta(q, par, l):
    yield U_loc(q, par, 0)
    for i in range(l):
        yield U_ent(q)
        yield U_loc(q, par, i+1)

#Set up input and output qubits.
qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]
q0, q1, q2 = [cirq.GridQubit(i, 0) for i in range(3)]

c = cirq.Circuit()
#c.append(U_phi(qubits,data_point))
#c.append(U_ent(qubits))
#c.append(U_theta_mt(qubits[0],np.array([1,2])))

c.append(U_data(qubits, data_point))
c.append(W_theta(qubits, par, 2))


simulator = cirq.Simulator()
circuit = cirq.Circuit()    
circuit.append(U_data(qubits, data_point))
circuit.append(W_theta(qubits, par, 2))
circuit.append(cirq.measure(*qubits, key='x'))
results = simulator.run(circuit, repetitions=3)
print(results.histogram(key='x'))


#print('Circuit:')
#print(c)







