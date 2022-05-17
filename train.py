import model
import numpy as np
import torch
import torchvision
import torchvision.datasets as datasets
from torchvision import transforms


transform = transforms.Compose([
    transforms.ToTensor()
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

training_loader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=0)

model = model.Net()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
loss_fn = torch.nn.MSELoss()


for x in range(100):

	running_loss = 0


	for i, data in enumerate(training_loader):
		#get the inputs and their labels
		inputs, labels = data

		#zero the gradients
		optimizer.zero_grad()

		#make predictions
		pred = model(inputs)

		#calculate loss
		loss = torch.nn.functional.nll_loss(pred, labels)
		loss.backward()

		running_loss += loss.item()

		optimizer.step()

	if x%10 == 0:
		print(running_loss)

