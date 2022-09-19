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
from torch import nn


class ConvNeuralNet(nn.Module):
	#  Determine what layers and their order in CNN object
	def __init__(self):
		super(ConvNeuralNet, self).__init__()
		self.model = torch.nn.Sequential(
			torch.nn.Conv2d(1, 4, (5, 5), 1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2),
			torch.nn.Conv2d(4, 12, (5, 5), 1),
			torch.nn.ReLU(),
			torch.nn.MaxPool2d(2),
			torch.nn.Conv2d(12, 10, (4, 4), 1)
		)

	# Progresses data across layers
	def forward(self, x):
		return self.model(x)


if __name__ == '__main__':
	batch_size = 5  # nombre de données lues à chaque fois
	nb_epochs = 10  # nombre de fois que la base de données sera lue
	eta = 0.001  # taux d'apprentissage

	# on lit les données
	((data_train, label_train), (data_test, label_test)) = torch.load(gzip.open('mnist.pkl.gz'))
	# on crée les lecteurs de données
	train_dataset = torch.utils.data.TensorDataset(data_train, label_train)
	test_dataset = torch.utils.data.TensorDataset(data_test, label_test)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)
	# on initialise le modèle et ses poids
	model = ConvNeuralNet()
	loss_func = torch.nn.MSELoss(reduction='sum')

	# Set optimizer with optimizer
	optimizer = torch.optim.SGD(model.parameters(), lr=eta)

	total_step = len(train_loader)

	# We use the pre-defined number of epochs to determine how many iterations to train the network on
	for epoch in range(nb_epochs):
		# Load in the data in batches using the train_loader object
		for x, t in train_loader:
			x = torch.reshape(x, (batch_size, 1, 28, 28))
			# Forward pass
			y = model(x)
			y = torch.reshape(y, (batch_size, 10))
			loss = loss_func(t, y)

			loss.backward()
			optimizer.step()
			optimizer.zero_grad()



		acc = 0.
		for x, t in test_loader:
			x = torch.reshape(x, (1, 1, 28, 28))
			y = model(x)
			acc += torch.argmax(y, 1) == torch.argmax(t, 1)
		print("Essai " + str(epoch + 1) + "/" + str(nb_epochs) + " : " + str(acc / data_test.shape[0]))
