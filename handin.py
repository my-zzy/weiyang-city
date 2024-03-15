import numpy as np
import pandas as pd

import torch
from torch.utils.data import TensorDataset, DataLoader
from my_model import MyModel

model = MyModel()
model.load_state_dict(torch.load('model_1.pth'))


# input = pd.read_csv('train.csv')
# mean = input.mean()
# std = input.std()
# # print(mean, std)
# normalized_input = (input - mean) / std

# print("input: ", input.head())
# # print(input.shape[0], input.shape[1])
# # print(input.info())
# # print(input.describe())
# # print(input.columns)


# train = normalized_input.iloc[0:900000, 4:15]
# train_label = input.iloc[0:900000, 1:3]  # notice that label should not be normalized
# # print("train: ", train.head())
# # print(train.shape[0], train.shape[1])
# # print(train_label.head())

# test = normalized_input.iloc[700001:900000, 4:15]
# test_label = input.iloc[700001:900000, 1:3]


# train_data = torch.tensor(train.values, dtype=torch.float32)

# train_label = torch.tensor(train_label.values, dtype=torch.float32)

# test_data = torch.tensor(test.values, dtype=torch.float32)

# test_label = torch.tensor(test_label.values, dtype=torch.float32)

# dear submission:
submit = pd.read_csv('test.csv')
mean_s = submit.mean()
std_s = submit.std()
normalized_submit = (submit - mean_s) / std_s

submit_input = normalized_submit.iloc[:, 2:13]
submit_train = torch.tensor(submit_input.values, dtype=torch.float32)

# print(train_data[0])
# print(train_label[0])
# print(test_data[0])
# print(test_label[0])
batch_size = 100
# train_dataset = TensorDataset(train_data, train_label)
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_dataset = TensorDataset(test_data, test_label)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# submit_loader = DataLoader(submit_train, batch_size=batch_size, shuffle=False)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


model.eval()
with torch.no_grad():
    print(1)
    submit_train = submit_train.to(device)
    predictions = model(submit_train)

print(2)
predictions = predictions.cpu()

indices = torch.arange(1, predictions.size(0) + 1).unsqueeze(1)

results = torch.cat((indices, predictions), dim=1)

# results = [(i + 1, prediction) for i, prediction in enumerate(predictions)]
print(results)
print(results.shape)

df = pd.DataFrame(results, columns=['ID', 'longitude_user', 'latitude_user'])
df.to_csv('submission_1.csv', index=False)