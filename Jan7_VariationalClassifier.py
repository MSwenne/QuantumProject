# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 11:46:20 2020

@author: vshas
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 10:49:07 2020

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

# Initialisation of some parameters
np.random.seed(239)     # Random seed
nr_qubits = 3           # Number of qubits
nr_layers = 4           # Number of layers
batch_size = 100        # Number of datapoints per training batch
shots = 2000            # Number of shots per datapoint
iterations = 25         # Number of iterations
key = ""                # String that contains all qubit-keynames

for i in range(nr_qubits):
    key += str(i)
        
# U_phi gate needed for fancy_U
def U_phi(q, W):
    # Apply rotation on every qubit based on datapoint
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent=W[i]/pi) 
        yield rot(q[i])
    # Apply controlled-rotation on every qubit-pair based on datapoint
    for i in range(len(q)-1):
        for j in range(i+1,len(q)):
            rot = cirq.ZPowGate(exponent=((pi-W[i])*(pi-W[j]))/pi)
            yield rot.on(q[j]).controlled_by(q[i])

# U_phi initialises the qubits in a state based on the current datapoint
def fancy_U(q, W):
    # Apply a Hadamard on every qubit
    for i in range(len(q)):
        yield cirq.H(q[i])
    # Apply U_phi
    yield U_phi(q, W)
    # Apply a Hadamard on every qubit
    for i in range(len(q)):
        yield cirq.H(q[i])
    # Apply U_phi
    yield U_phi(q, W)

# W_theta_p applies one layer of W_theta
def W_theta_p(q, theta):
    # Apply a controlled-Z gate on every qubit pair (i,i+1) based on theta
    for i in range(len(q)):
        yield cirq.CZ.on(q[(i+1)%len(q)],q[i])
    # Apply a Y and Z rotation on every qubit based on theta
    for i in range(len(q)):
        rot_z = cirq.ZPowGate(exponent=theta[2*i]/pi)
        rot_y = cirq.Ry(theta[2*i+1])
        yield rot_z(q[i])
        yield rot_y(q[i])

# W_theta applies a mapping from the qubits in the state based on the current
# datapoint to a quantum state that, when measured, can be mapped to a label
def W_theta(q, theta, layers):
    # Apply a Y and Z rotation on every qubit based on theta
    for i in range(len(q)):
        rot = cirq.ZPowGate(exponent = theta[2*i]/pi)
        rot = cirq.Ry(theta[2*i + 1])
        yield rot(q[i])
        yield rot(q[i])
    # Apply "layers" amount of layers using W_theta_p
    for i in range(1,layers+1):
        yield W_theta_p(q, theta[range(6*(i),6*(i+1))])
        
# Measures all qubits
def measure(q):
    for i in range(len(q)):
        yield cirq.measure(q[i], key=str(i))
        
# Builds the variational circuit
def circuit(q, W, theta, layers):
    yield fancy_U(q,W)
    yield W_theta(q, theta, layers)
    yield measure(q)

# Returns the absolute loss of the predictions
def abs_loss(labels, predictions):
    loss = 0 
    pred = np.round(predictions)
    for l, p in zip(labels, pred):
        loss = loss + np.abs(l - p)
    loss = loss / len(labels)
    return loss

# Returns the squared loss of the predictions
def squared_loss(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        loss = loss + (l - p)**2
    loss = loss/ len(labels)
    return loss

# Returns the accuracy over the predicted labels of a dataset
def accuracy(labels, p_hat, b):
    est_label = assign_label(p_hat, b)
    err = (labels*est_label - 1)/-2
    acc = 1 - np.sum(err)/len(err)
    return acc


def accuracy_x(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

# Returns a probability of seeing label 1 for a datapoint
def probability_estimate(results, shots):
    counter = (results.multi_measurement_histogram(keys="012"))
    p_hold = 0
    for j in counter:
        if j.count(1) % 2 == 1:
            p_hold += counter[j] 
    return p_hold/shots

# TODO
def assign_label(p, b):
    labels = np.ones([len(p),])*-1
    for i in range(len(p)):
        if (p[i] > ((1 - p[i]) - b)):
            labels[i] = 1
    return labels    

# TODO
def R(y, probs, b):
    p = 1 - probs
    y = 2*y - 1
    loss = 0
    R = 200
    for k in range(len(y)):
        if y[k] == 1:
            x = (math.sqrt(R)*(.5 - (probs[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        else:
            x = (math.sqrt(R)*(.5 - (p[k] - y[k]*(b/2))))/math.sqrt(2*probs[k]*p[k])
        loss = loss + (1 / (1 + math.exp(-x)))
    loss = loss / len(probs)
    return loss

# TODO
def empirical_risk(labels, pred_1, b):
    pred_0 = 1 - pred_1
    #pm = 2*labels - 1
    loss = 0
    R = 200
    for k in range(len(labels)):
        if labels[k] == 1:
            x = (math.sqrt(R)*(.5 - (pred_1[k] - labels[k]*(b/2))))/(math.sqrt(2*pred_1[k]*pred_0[k])+1e-10)
        else:
            x = (math.sqrt(R)*(.5 - (pred_0[k] - labels[k]*(b/2))))/(math.sqrt(2*pred_1[k]*pred_0[k])+1e-10)
        loss = loss + (1 / (1 + math.exp(-x)))
    loss = loss / len(labels)
    return loss


# TODO
def J(theta, X, qubits, nr_layers, shots):
    simulator = cirq.Simulator()
    p = np.zeros([len(X),])
    for i in range(len(X)):
        c = cirq.Circuit()
        c.append(circuit(qubits, X[i], theta, nr_layers))
        results = simulator.run(c, repetitions=shots)
        p[i] = probability_estimate(results, shots)
    return p

def calibration(a, c, alpha, gamma, X_t, Y_t, qubits, nr_layers, shots):
    number_q = len(qubits)
    number_par = (number_q*2)*(nr_layers+1)
    theta_0 = np.ones([number_par,])
    theta_0 = np.append(0)
    stat = 25
    initial_c = c
    delta_obj = 0
    for i in range(stat):
       print(i)
       delta = 2 * np.random.randint(2, size = len(theta_0)) - 1
       p_plus = J(theta_0+initial_c*delta, X_t, qubits, nr_layers, shots)
       p_minus = J(theta_0+initial_c*delta, X_t, qubits, nr_layers, shots)
       loss_p = empirical_risk(Y_t, p_plus, theta_0[-1])
       loss_m = empirical_risk(Y_t, p_minus, theta_0[-1])
       delta_obj += np.absolute(loss_p - loss_m) / stat

    a_new = a * 2 / delta_obj * initial_c
    return a_new    

# Helpful function that shows a progress bar
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

# Main function which runs the variational classifier
def main():
    # Set up qubit register
    qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]
    
    # Load the data and split parameters and labels
    df = pd.read_csv("QA_data_2pi.csv")
    X = df.iloc[:,:3].to_numpy()
    Y = df.iloc[:,3].to_numpy()
    Y = 2*Y - 1 
    
    # Initialise training data
    rows = random.sample(list(enumerate(X)),int(np.round((4/5)*len(X)))) #  Get 80% of the data as training data
    i_t = [x[0] for x in rows]
    X_t = X[i_t]
    Y_t = Y[i_t]
    i_s = [i for i in range(len(X)) if i not in i_t]
    X_s = X[i_s]
    Y_s = Y[i_s]
                                           
    # Initialise theta
    nr_par = (nr_qubits*2)*(nr_layers+1)
    init_theta = np.random.rand(nr_par,)*(2*pi)
    b = 0
    theta = np.append(init_theta,b)
    
    # Initialization
    a = 2.5
    c = 0.1
    z = a/6
    alpha = 0.602
    gamma = 0.101
        
    grad = np.zeros(nr_par)
    total_loss = np.zeros(iterations)
        
    # Start progress bar
    start = time.time()
    printProgressBar(0, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # Start iterations
    for k in range(1,iterations+1):
        # Update parameters
        c_n = c/(k**(gamma))
        a_n = a/(k**(alpha))
        z_n = z/(k**(alpha))
        delta_n = 2*np.random.randint(2, size = nr_par+1) - 1
    
        # Run variational classifier with theta+delta and theta-delta
        p_plus  = J(theta+c_n*delta_n, X_t, qubits, nr_layers, shots)
        p_minus = J(theta-c_n*delta_n, X_t, qubits, nr_layers, shots)
        # Calculate the loss for each run
        loss_plus  = empirical_risk(Y_t, p_plus, theta[-1]) 
        loss_minus = empirical_risk(Y_t, p_minus, theta[-1])
    
        # Compute gradient and update theta accordingly
        grad = ((loss_plus - loss_minus)/(2*c_n))/delta_n
        theta[1:-1] = (theta[1:-1] - a_n*grad[1:-1]) #% (2*pi)
        theta[-1] = (theta[-1] - z_n*grad[-1]) 
        # parameter b is probably taking too large steps.
        
        # Save average loss for plotting
        total_loss[k-1] = (loss_plus + loss_minus)/2
        # Add step to progress bar
        printProgressBar(k, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
        
    # Finish progress bar
    printProgressBar(iterations, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    end = time.time()    
    # Print time taken for all iterations
    print(end - start)
    
    # Plot average loss per iteration over all iterations
    fig = plt.figure(figsize=(15,10))
    plt.plot(range(1,iterations+1), total_loss, 'g-', markersize=2)
    # Nodig Tot_Loss, seed, eind_theta
    
    # train accuracy 
    p_t= J(theta, X_t, qubits, nr_layers, shots)
    acc_t = accuracy(Y_t, p_t, theta[-1])
    
    # test accuracy
    p_s = J(theta, X_s, qubits, nr_layers, shots)
    acc_s = accuracy(Y_s, p_s, theta[-1])
    print("train accuracy: ", acc_t)
    print("test accuracy:", acc_s)
    
# Start main
main()