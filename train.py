import model
import sys
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)


training_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers = 0)

model = model.Net()
model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.MSELoss()

#training loop
for x in range(30):

	running_loss = 0


	for i, data in enumerate(training_loader):
		#get the inputs and their labels
		inputs, labels = data
		#move to gpu
		inputs, labels = inputs.to(device), labels.to(device)


		#zero the gradients
		optimizer.zero_grad()

		#make predictions
		pred = model(inputs)

		#calculate loss
		loss = torch.nn.functional.nll_loss(pred, labels)
		loss.backward()

		running_loss += loss.item()

		optimizer.step()

	print("epoch number: " + str(x))
	print(running_loss)
	sys.stdout.flush()

