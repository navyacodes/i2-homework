import random
import numpy as np
import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image
#----------------------------------------------------------------#

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = tv.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

test_set = tv.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print(len(train_set))
print(len(test_set))
print(train_set[0][0].numpy().shape)
print(test_set[0][0].numpy().shape)

batch_size = 20
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

num_to_viz = 3
toPilImage = transforms.ToPILImage()

for i in range(num_to_viz):
    j = random.randrange(50000)
    # unnormalize image
    image_to_viz = toPilImage(train_set[j][0] / 2 + 0.5)
    image_to_viz_label = train_set[j][1]
    plt.imshow(image_to_viz)
    plt.title(classes[image_to_viz_label])
    plt.show()
  
class CIFAR10_CNN(nn.Module):
  def __init__(self, input_channels, output_size):
    super(CIFAR10_CNN, self).__init__()

    self.conv_L1 = nn.Conv2d(in_channels=input_channels, out_channels=10, kernel_size=(3,3), stride=1, padding=2)
    self.conv_L2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=(3,3), stride=1, padding=2)
    self.conv_L3 = nn.Conv2d(in_channels=20, out_channels=40, kernel_size=(3,3), stride=1, padding = 2)

    self.ff1 = nn.Linear(1000, 500)
    self.ff2 = nn.Linear(500, 250)
    self.ff3 = nn.Linear(250, output_size)

    self.maxpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)

    self.relu = nn.ReLU()


  def forward(self, input):
        x = self.conv_L1(input)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv_L2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv_L3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, 1)


        x = self.ff1(x)
        x = self.relu(x)
        x = self.ff2(x)
        x = self.relu(x)
        x = self.ff3(x)

        return x


f_model = CIFAR10_CNN(3, 10)

#----------------------------------------------------------------#
loss_func = nn.CrossEntropyLoss() # Mean Squared Error
optimizer = torch.optim.Adam(f_model.parameters(), lr=0.0007) # Adaptive Optimizer
#----------------------------------------------------------------#

num_epochs = 3;
n_total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
         # Forward pass
         outputs = f_model(images)
         loss = loss_func(outputs, labels)
         # Backward and optimize
         optimizer.zero_grad()
         loss.backward()
         optimizer.step()

         if (i+1) % 100 == 0:
             print (f'Epoch [{epoch+1}/{num_epochs}], Step[{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')

#----------------------------------------------------------------#
with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        outputs = f_model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs.data, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network on the 10000 test images: {acc} %')
#----------------------------------------------------------------#
