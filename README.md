# QuantumProject (Parameterized quantum circuits for supervised learning)
This project is about using parametrized quantum circuits (i.e., circuits where the gates depend on a set of parameters) as learning models in a supervised learning setting.
Literature:

https://www.nature.com/articles/s41586-019-0980-2
https://github.com/Qiskit/qiskit-community-tutorials/blob/master/artificial_intelligence/vqc.ipynb
https://pennylane.ai/qml/app/tutorial_variational_classifier.html

Objectives and tasks:
	– Learn and understand the parametrized quantum circuit classifier.
	– What are the parametrized quantum circuits that are being used as the classifier?
	– How do we train the parametrized quantum circuit?
	– How do we use a (trained) parametrized quantum circuit to classify previously unseen data?
	– What are the strengths and weaknesses of this classifier?

Provide an implementation of the parametrized quantum circuit classifier.
	1. Implement using Qiskit (the implementation itself is easier, however theruntimes are longer and therefore you can only run smaller experiments)
	2. Implement using Cirq (the implementation itself is harder, however theruntimes are faster and therefore you can run larger experiments)
	3. Implement using PennyLane.

Train and test the parametrized quantum circuit classifier on some of the following datasets:
	– Artificial datasets such as the datasets in the original paper, the bars-and-stripes datasets, the half-moon datasets or others that test properties of the classifiers.
	– Real-world datasets.
	– Your choice of dataset (explain why you thought this was a good datasetfor this classifier).
The report should contain:
	1. Description of the parametrized quantum circuits and how to use them as classifiers.
	2. Explanation of how to train the parametrized quantum circuits.
	3. Analysis of why this could be a good or interesting learning model.
	4. Description of your implementation and overview of the obtained results from the experiments.
	5. Discussion of the experiments and applicability of the learning model.

For a 10 and a paper:
	derive a learning setting in which you can show that the parametrized quantum circuit classifier has a provable advantage over standard (deep) neural networks with a fixed number of parameters