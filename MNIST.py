import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time

# convert image files to tensors of 4 dimensions (# of images, height, width, color channel)
transform = transforms.ToTensor()

# train data
train_data = datasets.MNIST(root='/cnn_data', train=True, download=True, transform=transform)

# test data
test_data = datasets.MNIST(root='/cnn_data', train=False, download=True, transform=transform)

# create small batches for images
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)
test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

conv1 = nn.Conv2d(1, 6, 3, 1)
conv2 = nn.Conv2d(6, 16, 3, 1)

class ConvolutionalNetwork(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv1 = nn.Conv2d(1,6,3,1)
    self.conv2 = nn.Conv2d(6,16,3,1)

    self.fc1 = nn.Linear(5*5*16, 120) # after testing we determine that the end result of two convolutions and two poolings is a 5x5 image. There are 16 of these.
    self.fc2 = nn.Linear(120, 84) # numbers are arbitrary, but trend down toward 10 (the number of categories we have [each of the ten digits])
    self.fc3 = nn.Linear(84, 10) # input = output of the first fully connected layer. output should match number of categories

  def forward(self, X):
    X = F.relu(self.conv1(X))
    X = F.max_pool2d(X,2,2)
    X = F.relu(self.conv2(X))
    X = F.max_pool2d(X,2,2)

    X = X.view(-1, 16*5*5) # -1 so we can vary the batch size

    X = F.relu(self.fc1(X))
    X = F.relu(self.fc2(X))
    X = self.fc3(X)

    return F.log_softmax(X, dim=1)


torch.manual_seed(41)

model = ConvolutionalNetwork()

# loss function optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001) # low learning rate = longer training time

start_time = time.time()

# create tracking variables
epochs = 5
train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
  trn_corr = 0
  tst_corr = 0

  for b,(X_train, y_train) in enumerate(train_loader):
    b += 1 # start batches at 1
    y_pred = model(X_train) # get predicted values. Not flattened
    loss = criterion(y_pred, y_train) # compare predictions to correct values

    predicted = torch.max(y_pred.data, 1)[1]
    batch_corr = (predicted == y_train).sum()
    trn_corr += batch_corr

    # update parameters
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if b%600 == 0:
      print(f"Epoch: {i} Batch: {b} Loss {loss.item()}")

  train_losses.append(loss)
  train_correct.append(trn_corr)


  with torch.no_grad(): # no gradient decent sowe don't update our model's weights and biases with test data
    for b,(X_test, y_test) in enumerate(test_loader):
      y_val = model(X_test)
      predicted = torch.max(y_val.data, 1)[1]
      tst_corr += (predicted == y_test).sum()

  loss = criterion(y_val, y_test)
  test_losses.append(loss)
  test_correct.append(tst_corr)

current_time = time.time()
total = current_time - start_time
print(f"Training took: {total/60} minutes!")

train_losses = [tl.item() for tl in train_losses]
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Validation Loss')
plt.title('Loss at Epoch')
plt.legend()


plt.plot([t/600 for t in train_correct], label="Training Accuracy")
plt.plot([t/100 for t in test_correct], label="Validation Accuracy")
plt.title("Accuracy at End of Each Epoch")
plt.legend()

test_load_everything = DataLoader(test_data, batch_size=10000, shuffle=False)

with torch.no_grad():
  correct = 0
  for X_test, y_test in test_load_everything:
    y_val = model(X_test)
    predicted = torch.max(y_val, 1)[1]
    correct += (predicted == y_test).sum()

print(f"{round(correct.item()/len(test_data)*100, 2)}% accurate!")

# grab an image
test_data[4143]

# grab just the data (no label)
test_data[4143][0]

# reshape it
test_data[4143][0].reshape(28,28)

# show image
plt.imshow(test_data[4143][0].reshape(28,28))

model.eval()
with torch.no_grad():
  new_prediction = model(test_data[4143][0].view(1,1,28,28))

new_prediction.argmax()
