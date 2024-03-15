import math
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

input = pd.read_csv('train.csv')
mean = input.mean()
std = input.std()
# print(mean, std)
normalized_input = (input - mean) / std

print("input: ", input.head())
# print(input.shape[0], input.shape[1])
# print(input.info())
# print(input.describe())
# print(input.columns)


train = normalized_input.iloc[0:700000, 4:15]
train_label = input.iloc[0:700000, 1:3]  # notice that label should not be normalized
# print("train: ", train.head())
# print(train.shape[0], train.shape[1])
# print(train_label.head())

test = normalized_input.iloc[700001:900000, 4:15]
test_label = input.iloc[700001:900000, 1:3]


train_data = torch.tensor(train.values, dtype=torch.float32)

train_label = torch.tensor(train_label.values, dtype=torch.float32)

test_data = torch.tensor(test.values, dtype=torch.float32)

test_label = torch.tensor(test_label.values, dtype=torch.float32)

# print(train_data[0])
# print(train_label[0])
# print(test_data[0])
# print(test_label[0])
batch_size = 100
train_dataset = TensorDataset(train_data, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = TensorDataset(test_data, test_label)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def distance(x1, x2):

    x1 = np.radians(x1)
    x2 = np.radians(x2)
    lat1, lon1 = x1[:, 1], x1[:, 0]
    lat2, lon2 = x2[:, 1], x2[:, 0]
    # lat1 = math.radians(lat1)
    # lon1 = math.radians(lon1)
    # lat2 = math.radians(lat2)
    # lon2 = math.radians(lon2)

    # Haversine Equation
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius = 6371  # km
    distance = radius * c

    return distance


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc_1 = nn.Linear(11, 16)
        self.fc_2 = nn.Linear(16, 8)
        self.fc_3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn_16 = nn.BatchNorm1d(16)
        self.bn_8 = nn.BatchNorm1d(8)

    def forward(self, x):
        x = self.fc_1(x)
        x = self.bn_16(x)
        x = self.relu(x)
        x = self.fc_2(x)
        x = self.bn_8(x)
        x = self.relu(x)
        out = self.fc_3(x)
        return out

    
model = MyModel()

# Manually setting the weights for each layer
# todo


# criterion = torch.nn.CrossEntropyLoss()
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.0001)

num_epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_count = 0

for epoch in range(num_epochs):
    running_loss = 0.0
    
    model.train()
    for inputs, labels in train_loader:
        # print(inputs.shape)
        # print(inputs[0])
        # print(labels.shape)
        # print(labels[0])
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # loss
        running_loss += loss.item()

    # print(labels.size(0))
    # print(inputs.size(0))
    # print(outputs[:3])
    # print("loss one time ",loss.item()) 
    epoch_loss = running_loss / labels.size(0)
    # print(f"Epoch {epoch+1}, Loss: {epoch_loss:.4f}")

    # begin testing
    model.eval()
    total_correct = 0
    total_samples = 0
    distance_save = np.array([])

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # forward
            outputs = model(inputs)

            # accuracy
            if outputs.shape == labels.shape:
                # print(outputs.shape)
                distances = distance(outputs.cpu(), labels.cpu())   # Use Tensor.cpu() to copy the tensor to host memory first.
                count = torch.sum(distances < 50/1000).item()
            else:
                print("wrong shape")

            total_correct += count
            total_samples += labels.size(0)
            distance_save = np.append(distance_save, distances)

        # print("count ", count)
        # print(labels.size(0))
        print(outputs)
        print(distance_save)

    test_accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Count: {total_correct}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Test Accuracy: {test_accuracy:.8f}")

    if total_correct > best_count:
        torch.save(model.state_dict(), 'model_1.pth')