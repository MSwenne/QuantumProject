"""
Parameterized quantum circuits for supervised learning

@author: Vince Hasse
@author: Martijn Swenne
Last edited:
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from math import pi
import numpy as np
import random
import sympy
import math
import cirq
import time
import os

# Initialisation of some parameters
seed = 239              # Random seed
np.random.seed(seed)    # Initialise random seed
nr_qubits = 3           # Number of qubits
nr_layers = 4           # Number of layers
batch_size = 10        # Number of datapoints per training batch
shots = 2            # Number of shots per datapoint
iterations = 1         # Number of iterations
file = "data{}.txt"     # Base filename for writing away data
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
def accuracy(labels, predictions):
    loss = 0
    for l, p in zip(labels, predictions):
        if abs(l - p) < 1e-5:
            loss = loss + 1
    loss = loss / len(labels)
    return loss

# Returns a probability of seeing label 1 for a datapoint
def probability_estimate(results):
    counter = (results.multi_measurement_histogram(keys="012"))
    p_hold = 0
    for j in counter:
        if j.count(1) % 2 == 1:
            p_hold += counter[j] 
    return p_hold/shots

# TODO
def assign_label(p, Y, b):
    Y_pm = 2 * Y - 1
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
def J_w(theta, X, qubits, nr_layers, shots):
    simulator = cirq.Simulator()
    p = np.zeros([len(X),])
    for i in range(len(X)):
        p_hold =0
        c = cirq.Circuit()
        c.append(circuit(qubits, X[i], theta, nr_layers))
        results = simulator.run(c, repetitions=shots)
        probability_estimate(results)
        p[i] = p_hold/shots
    return p

def calibration():
    stat = 25
    hold_c0 = parameters[0]
    initial_c = parameters[1]
    delta_obj = 0
    for i in range(stat):
       print(i)
       delta = 2 * np.random.randint(2, size = len(theta)) - 1
       obj_plus = J_w(theta+initial_c*delta, X_t, qubits, nr_layers, shots)
       obj_minus = J_w(theta+initial_c*delta, X_t, qubits, nr_layers, shots)
       loss_p = squared_loss(Y_t, obj_plus)
       loss_m = squared_loss(Y_t, obj_minus)
       delta_obj += np.absolute(loss_p - loss_m) / stat

    #c_new = hold_c0 * 2 / delta_obj * initial_c    

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

# Reads the data from a file
def read_from_file(filename):
    f = open(filename,"r")
    f.readline()
    nr_qubits  = int(f.readline().split(" ")[-1])
    nr_layers  = int(f.readline().split(" ")[-1])
    batch_size = int(f.readline().split(" ")[-1])
    shots      = int(f.readline().split(" ")[-1])
    iterations = int(f.readline().split(" ")[-1])
    seed       = int(f.readline().split(" ")[-1])
    Tot_Loss = []
    f.readline()
    f.readline()
    for i in range(iterations):
        Tot_Loss.append(float(f.readline().strip()))
    theta = []
    f.readline()
    for i in range((nr_qubits*2)*(nr_layers+1)):
        theta.append(float(f.readline().strip()))
    return nr_qubits, nr_layers, batch_size, shots, iterations, seed, Tot_Loss, theta

# Writes data to a file
def write_to_file(nr_qubits, nr_layers, batch_size, shots, iterations, seed, Tot_Loss, theta):
    # Open new file
    counter = 0
    filename = file
    while os.path.isfile(filename.format(counter)):
        counter += 1
    filename = filename.format(counter)
    f = open(filename,"w+")
    # Write Params, Tot_Loss and end_theta to file
    f.write("RUN PARAMS:\n")
    f.write("\tnr_qubits:  %d\n" % nr_qubits)
    f.write("\tnr_layers:  %d\n" % nr_layers)
    f.write("\tbatch_size: %d\n" % batch_size)
    f.write("\tshots:      %d\n" % shots)
    f.write("\titerations: %d\n" % iterations)
    f.write("\tseed:       %d\n" % seed)
    f.write("\tRESULTS:\n")
    f.write("\t\tTot_Loss:\n\t\t\t")
    f.write("\n\t\t\t".join(str(elem) for elem in Tot_Loss))
    f.write("\n\t\teind_theta:\n\t\t\t")
    f.write("\n\t\t\t".join(str(elem) for elem in theta))

# Main function which runs the variational classifier
def main():
    # Set up qubit register
    qubits = [cirq.GridQubit(i, 0) for i in range(nr_qubits)]

    # Load the data and split parameters and labels
    df = pd.read_csv("QA_data_x.csv")
    X = df.iloc[:,:3].to_numpy()
    Y = df.iloc[:,3].to_numpy()

    # Initialise training data
    rows = random.sample(range(len(X)),int(np.round((4/5)*len(X)))) # Get a percentage of the data as training data
    i_t = [x[0] for x in rows]                                      # Get the training indexes
    X_t = X[i_t]                                                    # Get the training parameters
    Y_t = Y[i_t]                                                    # Get the training labels
    i_s = [i for i in range(len(X)) if i not in i_t]                # Get the test indexes
    X_s = X[i_s]                                                    # Get the test parameters
    Y_s = Y[i_s]                                                    # Get the test labels
#     indexes = np.array(range(len(X)))                  # 
#     m = int(np.round((4/5)*len(X)))                    # 
#     train = np.random.choice(len(X), m, replace=False) # 
#     test = indexes[~np.isin(indexes,train)]            # 
#     X_t = X[train,:]                                   # 
#     Y_t = Y[train]                                     #

    # Initialise theta
    nr_par = (nr_qubits*2)*(nr_layers+1)
    init_theta = np.random.rand(nr_par,)*(2*pi)
    b = 0
    theta = np.append(init_theta,b)

    # Initialise classifier parameters
    eye = np.eye(nr_par)
    a = 2.5
    c = 0.1
    alpha = 0.602
    gamma = 0.101
    parameters = np.array([a,c,alpha,gamma])
    batch_ix = np.array(range(len(X_t)))
    plot_ix = 5
    P = int(iterations/plot_ix)
    tot_loss = np.zeros(P)
    Tot_Loss = np.zeros(iterations)
    z = a/pi
    loss_est = 0
    iw = 0
    
    # Start progress bar
    start = time.time()
    printProgressBar(0, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    # Start iterations
    for k in range(1,iterations+1):
        # Update parameters
        #batch_ix = np.random.randint(0, len(X_t), (batch_size,))
        c_n = c/(k**(gamma))
        a_n = a/(k**(alpha))
        z_n = z/(k**(alpha))
        gradient = np.zeros(nr_par)
        delta_n = 2*np.random.randint(2, size = nr_par+1) - 1

        # Run variational classifier with theta+delta and theta-delta
        p_plus  = J_w(theta+c_n*delta_n, X_t, qubits, nr_layers, shots)
        p_minus = J_w(theta-c_n*delta_n, X_t, qubits, nr_layers, shots)
        # Calculate the loss for each run
        loss_plus  = R(Y_t, p_plus, theta[-1]) 
        loss_minus = R(Y_t, p_minus, theta[-1])

        # Compute gradient and update theta accordingly
        grad = ((loss_plus - loss_minus)/(2*c_n))/delta_n
        theta[1:-1] = (theta[1:-1] - a_n*grad[1:-1]) #% (2*pi)
        theta[-1] = (theta[-1] - z_n*grad[-1]) 
        # parameter b is probably taking too large steps.

        # Save average loss for plotting
        Tot_Loss[k-1] = (loss_plus + loss_minus)/2
        # Add step to progress bar
        printProgressBar(k, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)

    # Finish progress bar
    printProgressBar(iterations, iterations, prefix = 'Progress:', suffix = 'Complete', length = 50)
    end = time.time()    
    # Print time taken for all iterations
    print(end - start)

    # Plot average loss per iteration over all iterations
    fig = plt.figure(figsize=(15,10))
    plt.plot(range(1,iterations+1), Tot_Loss, 'g-', markersize=2)

    write_to_file(nr_qubits, nr_layers, batch_size, shots, iterations, seed, Tot_Loss, theta)

# Start main
main()