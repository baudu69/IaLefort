# coding: utf8
# !/usr/bin/env python
# ------------------------------------------------------------------------
# Perceptron en pytorch (en utilisant les outils de Pytorch)
# Écrit par Mathieu Lefort
#
# Distribué sous licence BSD.
# ------------------------------------------------------------------------

import gzip

import torch
import numpy as np
from matplotlib import pyplot as plt


def init_model(input):
	model = torch.nn.Sequential()
	for i in range(0, len(input) - 2):
		model.append(torch.nn.Linear(input[i], input[i + 1]))
		model.append(torch.nn.ReLU())
	model.append(torch.nn.Linear(input[len(input) - 2], input[len(input) - 1]))
	return model


if __name__ == '__main__':
	batch_size = 10  # nombre de données lues à chaque fois
	nb_epochs = 10  # nombre de fois que la base de données sera lue
	eta = 0.001  # taux d'apprentissage
	hidden1 = 400
	hidden2 = 200

	# on lit les données
	((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
	# on initialise le modèle et ses poids
	model = init_model([data_train.shape[1], hidden1, hidden2, label_train.shape[1]])

	# on initialise l'optimiseur
	loss_func = torch.nn.MSELoss(reduction='sum')
	optimModel = torch.optim.SGD(model.parameters(), lr=eta)

	history = []

	for n in range(nb_epochs):
		# on lit toutes les données d'apprentissage
		for x, t in train_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on met à jour les poids
			loss = loss_func(t, y)
			loss.backward()
			optimModel.step()
			optimModel.zero_grad()

		# test du modèle (on évalue la progression pendant l'apprentissage)
		acc = 0.
		# on lit toutes les donnéees de test
		for x, t in test_loader:
			# on calcule la sortie du modèle
			y = model(x)
			# on regarde si la sortie est correcte
			acc += torch.argmax(y, 1) == torch.argmax(t, 1)
		# on affiche le pourcentage de bonnes réponses
		print("Essai " + str(n+1) + "/"+str(nb_epochs)+" : " + str(acc / data_test.shape[0]))
		history.append(float(acc / data_test.shape[0]))
		plt.plot(history)
		plt.axis([0, nb_epochs - 1, 0, 1])
		plt.suptitle(
			'DeepNetwork : eta=' + str(eta) + ', hidden1=' + str(hidden1) + ', hidden2=' + str(hidden2) + ', batchsize=' + str(batch_size))
		plt.show()
