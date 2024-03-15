import numpy as np
import pandas as pd
import torch


input = pd.read_csv('train.csv', dtype='float64')
para_df = pd.read_csv('parameter.csv')

input_for_search = torch.tensor(input[['sc_eci', 'nc_1_eci', 'nc_2_eci']].values, dtype=torch.float64)
para = torch.tensor(para_df[['eci', 'enb_latitude', 'enb_longitude']].values, dtype=torch.float64)
print('para', para.shape)
print(para[0])


mean = input.mean()
std = input.std()
# print(mean, std)
normalized_input = (input - mean) / std

normalized_input_save = normalized_input.iloc[0:700000, :]
save_list = ['sc_rsrp', 'nc_1_freq', 'nc_1_rsrp', 'nc_2_freq', 'nc_2_rsrp']
normalized_input_for_save = torch.tensor(normalized_input_save[save_list].values, dtype=torch.float64)

# magnification = 1000
# min = 122


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

train_data = torch.tensor(train.values, dtype=torch.float64)

train_label = torch.tensor(train_label.values, dtype=torch.float64)

test_data = torch.tensor(test.values, dtype=torch.float64)

test_label = torch.tensor(test_label.values, dtype=torch.float64)

# torch.save(train_data, 'train_data.pt')
torch.save(train_label, 'train_label.pt')
torch.save(test_data, 'test_data.pt')
torch.save(test_label, 'test_label.pt')


# add more data from parameter
# the higher dimension is at the beginning of the tuple
print(1)
print(train_data.shape)
print(train_data[0])



# change the label
train_label[:, 0] = (train_label[:, 0] - 109) * 1000
# print(train_label[:, 0])

train_label[:, 1] = (train_label[:, 1] - 24) * 1000

test_label[:, 0] = (test_label[:, 0] - 109) * 1000

test_label[:, 1] = (test_label[:, 1] - 24) * 1000

'''
what's in train tensor ?
original:
sc_eci  sc_enb_id  sc_rsrp   
nc_1_eci  nc_1_pci  nc_1_freq  nc_1_rsrp   
nc_2_eci  nc_2_pci  nc_2_freq  nc_2_rsrp
present:
enb_latitude  enb_longitude  (sc_enb_id)  sc_rsrp   
enb_latitude  enb_longitude   nc_1_freq  nc_1_rsrp 
enb_latitude  enb_longitude   nc_2_freq  nc_2_rsrp

'''
enb_0 = torch.zeros(700000, 2)
enb_1 = torch.zeros(700000, 2)
enb_2 = torch.zeros(700000, 2)
enb_list = [enb_0, enb_1, enb_2]
search_list = -1

for enb in enb_list:
    a = 0
    b = 0
    c = 0
    search_list = search_list + 1
    for i in range(700000):       # should be 700000
        search = input_for_search[i, search_list]   # search_list = [0,1,2]
        indices = torch.where(para == search)
        if (a+b+c)%1000==0:
            print(a,b,c)
        if indices[0].shape[0] == 0:
            # print(a)
            # print("no", a+b+c, search)
            a = a + 1
            # print("omg find nothing!")

        elif indices[0].shape[0] > 1:
            # print("omg find too many!")
            row = indices[0][0].item()
            col = indices[1][0].item()
            enb[i, 0] = para[row, 1]    # lat
            enb[i, 1] = para[row, 2]    # lon
            b = b + 1

        else:
            # print("find one")
            row = indices[0][0].item()
            col = indices[1][0].item()
            enb[i, 0] = para[row, 1]    # lat
            enb[i, 1] = para[row, 2]    # lon
            c = c + 1
        # row = indices[0][0]
        # col = indices[1][0]
        # enb_1[i, 0] = row
        # enb_1[i, 1] = col
        

    # print(torch.sum(enb_1 == 0).item())
    print("end", a, b, c)


# print(train_data[0])
# print(train_label[0])
# print(test_data[0])
# print(test_label[0])
    
tot = torch.cat((enb_0/10, enb_1/10, enb_2/10, normalized_input_for_save), dim=1)
print(2)
print(tot.shape)
print(tot[0])
torch.save(tot, 'train_opt.pt')
# mean = tot.mean(dim=0)
# std = tot.std(dim=0)

# normalized_tot = (tot - mean) / std
# print(3)
# print(normalized_tot.shape)
# print(normalized_tot[0])

print('verify the shape')
print(tot.shape)
print(train_data.shape)

# todo
# there are many zeros in 'tot' due to data missing