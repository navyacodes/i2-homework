import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
#----------------------------------------------------------------#

train_data = tv.datasets.MNIST('./data', train=True, transform=transforms.ToTensor(), download=True)
test_data = tv.datasets.MNIST('./data', train=False, transform=transforms.ToTensor(), download=True)
print("train data shape: ",  train_data.data.numpy().shape)
print("train labels shape: ", train_data.targets.numpy().shape)
print("test data shape: ", test_data.data.numpy().shape)
print("test labels shape: ", test_data.targets.numpy().shape)

batch_size = 60
train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

num_of_digits_to_viz = 3
for i in range(num_of_digits_to_viz):
    to_reshape = train_data.data.numpy()[i]
    plt.matshow(to_reshape.reshape(28, 28))
    plt.show()
    print(f"Associated Label: {train_data.targets.numpy()[i]}")

class MNIST_DNN(nn.Module):
    def __init__(self, input_layer_size, hidden_layer_A_size, hidden_layer_B_size, hidden_layer_C_size, output_layer_size):
        super(MNIST_DNN, self).__init__()

        self.input_layer_size = input_layer_size
        self.hidden_layer_A_size = hidden_layer_A_size
        self.hidden_layer_B_size = hidden_layer_B_size
        self.hidden_layer_C_size = hidden_layer_C_size
        self.output_layer_size = output_layer_size

        self.l1 = nn.Linear(self.input_layer_size, self.hidden_layer_A_size)
        self.l2 = nn.Linear(self.hidden_layer_A_size, self.hidden_layer_B_size)
        self.l3 = nn.Linear(self.hidden_layer_B_size, self.hidden_layer_C_size)
        self.l4 = nn.Linear(self.hidden_layer_C_size, self.output_layer_size)

        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax()

    def forward(self, input):
        x = self.l1(input)
        x = self.sigmoid(x)
        x = self.l2(x)
        x = self.sigmoid(x)
        x = self.l3(x)
        x = self.sigmoid(x)
        x = self.l4(x)
        output = self.softmax(x)
        return output

model = MNIST_DNN(784, 392, 196, 98, 10)

loss_func = nn.CrossEntropyLoss() # Mean Squared Error
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # Adaptive Optimizer

num_epochs = 3

n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
         # origin shape: [100, 1, 28, 28]
         # resized: [100, 784]
         images = images.reshape(-1, 28*28)
         # Forward pass
         outputs = model(images)
         loss = loss_func(outputs, labels)
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
