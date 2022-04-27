from multiprocessing.dummy import freeze_support

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.dataloader import DataLoader

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from PIL import Image


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #The first convolotional layer, (Input channels, output channels, Kernal size)
        self.conv1 = nn.Conv2d(4, 4, (3, 3))
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(4, 4, (3, 3))
        self.conv3 = nn.Conv2d(4, 4, (3, 3))
        self.fc1 = nn.Linear(100, 90)
        self.fc2 = nn.Linear(90, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def RGBA_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA')


def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

if __name__ == '__main__':
    net = Net()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print('There are %d GPU(s) available.' % torch.cuda.device_count())
        print('We will use the GPU:', torch.cuda.get_device_name(0))

    else:
        print('No GPU available, using the CPU instead.')

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    net.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0001)

    #data_dir = "PlayerTypesImages"
    #classes = ('Assault', 'Journey', 'Manage')

    data_dir = "ExObImages"
    classes = ('Exploration', 'Objective')

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5, 0.5), (0.5, 0.5, 0.5, 0.5))])
    #transform = transforms.ToTensor()
    dataset = ImageFolder(data_dir, transform=transform, loader=RGBA_loader)

    batch_size = 1
    val_size = round(len(dataset)*0.2)
    train_size = len(dataset) - val_size
    #
    #train_data, val_data = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    #val_dl = DataLoader(val_data, batch_size=batch_size, shuffle=True, num_workers=0)


    # transform = transforms.Compose(
    #     [transforms.ToTensor(),
    #      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    #
    # batch_size = 1
    #
    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
    #                                         download=True, transform=transform)
    # train_dl = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
    #                                           shuffle=True, num_workers=2)
    #
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False,
    #                                        download=True, transform=transform)
    # val_dl = torch.utils.data.DataLoader(testset, batch_size=batch_size,
    #                                          shuffle=False, num_workers=2)
    #
    # classes = ('plane', 'car', 'bird', 'cat',
    #            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get some random training images
    dataiter = iter(train_dl)
    images, labels = dataiter.next()

    # show images
    # imshow(torchvision.utils.make_grid(images))

    freeze_support()
    for epoch in range(1):  # loop over the dataset multiple times
        net.training = True
        running_loss = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            #print(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
    dataiter = iter(train_dl)
    images, labels = dataiter.next()
    #
    # #
    net = Net()
    net.load_state_dict(torch.load(PATH))
    outputs = net(images)
    _, predicted = torch.max(outputs,1)

    print(labels)
    print(predicted)
    #
    # print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'for j in range(1)))
    net.training = False
    net.eval()
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in train_dl:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            print("Predicted " + str(predicted) + "  label " + str(labels))
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in train_dl:
            images, labels = data
            outputs = net(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    print(correct_pred)
    print(total_pred)
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for class: {classname:5s} is {accuracy:.1f} %')
