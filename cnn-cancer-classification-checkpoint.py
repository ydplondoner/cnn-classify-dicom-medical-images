import argparse
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms
from torch.utils.data import random_split
from torch.utils.data.dataset import Subset
from torchsummary import summary
from torchvision.datasets import ImageFolder
from typing import List
import time


# Deal with 32x32 images for this task
def img_loader(filename):
    a = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    return cv2.resize(a, (32,32))


def load_data(data_dir: str, train_test_split: List[float]) -> (Subset, Subset, List[str]):
    """
    :param data_dir: input directory, hierarchy by class
    :param train_test_split: List of two floats [training_data, test_data], should have a sum of 1.0
    :return: Tuple of (training data, test data, list of classes)
    """
    transform = transforms.Compose([transforms.ToTensor()])
    image_data = ImageFolder(root=data_dir, loader=img_loader, transform=transform)
    train_data, test_data=random_split(image_data, train_test_split)
    return train_data, test_data, image_data.classes


class Net(nn.Module):
    def __init__(self, name=None, classes=None):
        super(Net, self).__init__()
        self.name = name
        self.classes = classes

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(10, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 16)
        self.fc3 = nn.Linear(16, 4)

        # compute the total number of parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self.name + ': total params:', total_params)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(train_data: Subset,num_epochs: int, learning_rate: int,
          momentum: int, batch_size: int, net: nn.Module, checkpoint_path):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)

    start = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels.reshape(-1))

            optimizer.zero_grad()  # zero the parameter gradients
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 50 == 49:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 49))
                running_loss = 0.0

    print('Finished Training')
    end = time.time()
    print('training time ', end - start)
    print('Saving checkpoint')
    model_checkpoint = {
        'epoch': num_epochs,
        'classes': net.classes,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': running_loss,
        }

    print(f"Saving checkpoint in dir {checkpoint_path}")
    torch.save(model_checkpoint, checkpoint_path)
    print("Completed saving checkpoint")


# Print accuracy of overall network and indvidual classes
def test(test_data: Subset, net: nn.Module, classes: List[str]):
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=4, shuffle=True)
    correct = 0
    total = 0
    cnt = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            cnt += 1

    print(f"accuracy = {100 * correct / total:.2f}% for network on {cnt} test images")

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            if c.shape > torch.Size([]):
                for i in range(len(c)):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

    for i in range(len(classes)):
        print('Accuracy of %5s : %2d %%' % (
            classes[i], 100 * class_correct[i] / class_total[i]))


def evaluate_face(checkpoint_dir: str, faces_to_evaluate_dir: str, classes: List[str]):
    net = Net(name='LetNet5EvalOnly')
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(checkpoint_dir)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    classes = checkpoint['classes']
    net.eval()

    _, data_to_eval, _ = load_data(faces_to_evaluate_dir, [0.0, 1.0])
    eval_data_loader = torch.utils.data.DataLoader(data_to_eval, batch_size=1, shuffle=False)

    with torch.no_grad():
        for i, data in enumerate(eval_data_loader):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)

            print("T2 What's the facial expression in the picture?")
            print(f"{data_to_eval.dataset.imgs[i][0]} has expression:")
            print(classes[predicted])


def main():
    # Model hyperparameters
    num_of_epochs = 100
    learning_rate = 0.001
    momentum = 0.9
    batch_size = 4
    checkpoint_path = "./checkpoints/facial_expression_model.pt"
    eval_dir = "./data/faces_data/faces_to_eval/"
    data_dir = "./data/faces_data/faces4_by_expression/"

    parser = argparse.ArgumentParser(description='Facial expression NN')
    parser.add_argument('--evaluate_only', default=False, action='store_true')
    args = parser.parse_args()

    if not args.evaluate_only:
        train_test_split = [0.8, 0.2]
        train_data, test_data, classes = load_data(data_dir, train_test_split)
        net = Net(name='LetNet5', classes=classes)

        print("classes are: ", classes)
        train(train_data, num_of_epochs, learning_rate, momentum, batch_size, net, checkpoint_path)
        test(test_data, net, classes)

    evaluate_face(checkpoint_path, eval_dir, None)


if __name__ == '__main__':
    main()