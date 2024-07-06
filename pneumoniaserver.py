import numpy as np
import torch 
from pathlib import Path
import sklearn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from tqdm import tqdm
import sklearn.metrics 
import os
import pickle




class PneumaniaTest(Dataset):
    def __init__(self) -> None:
        positive_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/test/PNEUMONIA')
        negative_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/test/NORMAL')
        x =[]
        y = []
        for item in positive_path.iterdir():
            img = Image.open(item)
            y.append([0, 1])
            if img.mode!="L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        for item1 in negative_path.iterdir():
            img = Image.open(item1)
            y.append([1, 0])
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        self.testinputtensor = torch.tensor(x, dtype=torch.float)
        self.testoutputtensor = torch.tensor(y, dtype=torch.float)


    def __getitem__(self, index):
        return self.testinputtensor[index, :, :].unsqueeze(0), self.testoutputtensor[index, :]
    
    def __len__(self):
        return self.testinputtensor.size(dim=0)
    

class PneumaniaTrain(Dataset):
    def __init__(self) -> None:
        positive_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/train/PNEUMONIA')
        negative_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/train/NORMAL')
        x =[]
        y = []
        for item in positive_path.iterdir():
            img = Image.open(item)
            y.append([0, 1])
            if img.mode!="L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        for item1 in negative_path.iterdir():
            img = Image.open(item1)
            y.append([1, 0])
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        self.testinputtensor = torch.tensor(x, dtype=torch.float)
        self.testoutputtensor = torch.tensor(y, dtype= torch.float)


    def __getitem__(self, index):
        return self.testinputtensor[index, :, :].unsqueeze(0), self.testoutputtensor[index, :]
    
    def __len__(self):
        return self.testinputtensor.size(dim=0)


class PneumaniaVal(Dataset):
    def __init__(self) -> None:
        positive_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/val/PNEUMONIA')
        negative_path = Path(r'/storage/brno2/home/yauheni/pneumania_illness/val/NORMAL')
        x =[]
        y = []
        for item in positive_path.iterdir():
            img = Image.open(item)
            y.append([0, 1])
            if img.mode!="L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        for item1 in negative_path.iterdir():
            img = Image.open(item1)
            y.append([1, 0])
            if img.mode != "L":
                img = img.convert("L")
            img = img.resize((224, 224))
            x.append(np.array(img))
        self.testinputtensor = torch.tensor(x, dtype=torch.float)
        self.testoutputtensor = torch.tensor(y, dtype=torch.float)


    def __getitem__(self, index):
        return self.testinputtensor[index, :, :].unsqueeze(0), self.testoutputtensor[index, :]
    
    def __len__(self):
        return self.testinputtensor.size(dim=0)
    




#implementation of ResNet

class Block(nn.Module):
    def __init__(self, input_channels, out_channals, identity_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expension = 4
        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=out_channals, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channals)
        self.conv2 = nn.Conv2d(in_channels=out_channals, out_channels=out_channals, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channals)
        self.conv3 = nn.Conv2d(in_channels=out_channals, out_channels=out_channals*self.expension, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channals*self.expension)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample


    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes, image_channels):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, layers[0], out_channels=64, stride=1)
        self.layer2 = self.make_layer(block, layers[1], out_channels=128, stride=2)
        self.layer3 = self.make_layer(block, layers[2], out_channels=256, stride=2)
        self.layer4 = self.make_layer(block, layers[3], out_channels=512, stride=2)
        self.avgpooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*4, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpooling(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x

    def make_layer(self, block, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers =[]
        if stride != 1 or self.in_channels != out_channels*4:
            identity_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride),
                                                nn.BatchNorm2d(out_channels*4))
        layers.append(block(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels*4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)


def ResNet50(img_channels=1, num_classes=2):
    return ResNet(Block, [3, 4, 6, 3], num_classes, img_channels)



root_dir = r"/storage/brno2/home/yauheni/pneumania_illness"
loss_fn = nn.CrossEntropyLoss()
lr = 1e-4
num_epochs = 100
model = ResNet50()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimazer = torch.optim.Adam(model.parameters())
datatrain = PneumaniaTrain()
datatest = PneumaniaTest()
dataval = PneumaniaVal()
dataloader_train = DataLoader(dataset=datatrain, batch_size=64, shuffle=True)
dataloader_val = DataLoader(dataset=dataval, batch_size=64, shuffle=True)
dataloader_test = DataLoader(dataset=datatest, batch_size=64, shuffle=True)
history = []
history1 = []
acc_val = []
acc_test = []
acc_train = []
max_acc = 0 
best_score = 0
for epochs in range(num_epochs):
    with tqdm(total=81, desc=f'Epoch {epochs + 1}/{num_epochs}', unit='batch') as pbar:
        running_loss = 0
        val_loss = 0
        model.train()
        for i, data in enumerate(dataloader_train):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimazer.zero_grad()
            output = model(inputs)
            loss = loss_fn(output, labels)
            loss.backward()
            optimazer.step()
            running_loss += loss.item()
            
            labels_cpu = labels.cpu().numpy()
            output_cpu = output.detach().cpu().numpy()

            # Используем np.argmax для вычисления меток
            labels_for_acc = np.argmax(labels_cpu, axis=1)
            output_for_acc = np.argmax(output_cpu, axis=1)

            # Вычисляем точность
            acc = sklearn.metrics.accuracy_score(labels_for_acc, output_for_acc)
            acc_train.append(acc)
            pbar.update(1)
        history.append(running_loss)
    model.eval()
    for j, data_eval in enumerate(dataloader_val):
        with torch.no_grad():
            inputs_val, labels_val = data_eval
            inputs_val = inputs_val.to(device)
            labels_val = labels_val.to(device)
            output_val = model(inputs_val)
            loss = loss_fn(output_val, labels_val)
            val_loss += loss.item()

            labels_cpu = labels_val.cpu().numpy()
            output_cpu = output_val.detach().cpu().numpy()

            # Используем np.argmax для вычисления меток
            labels_for_acc = np.argmax(labels_cpu, axis=1)
            output_for_acc = np.argmax(output_cpu, axis=1)

            acc = sklearn.metrics.accuracy_score(labels_for_acc, output_for_acc)
            acc_val.append(acc)

            if best_score<acc:
                torch.save(model.state_dict(), os.path.join(root_dir, "best_metric_model_pneumania.pth"))
            

        history1.append(val_loss)
correct = 0
total = 0
model.load_state_dict(torch.load(os.path.join(root_dir, "best_metric_model_pneumania.pth")))
model.to(device)
for j, data in enumerate(dataloader_test):
    with torch.no_grad():
        inputs_for_test, labels_for_test = data
        inputs_for_test = inputs_for_test.to(device)
        labels_for_test = labels_for_test.to(device)
        output_for_test = model(inputs_for_test)
        
        _, predicted = torch.max(output_for_test, 1)

        # Преобразование one-hot encoded меток в индексы классов
        true_labels = torch.argmax(labels_for_test, 1)

        # Сравнение предсказаний с истинными метками
        correct += (predicted == true_labels).sum().item()
        total += labels_for_test.size(0)



file_path = r'/storage/brno2/home/yauheni/pneumania_illness/history_train_pneumania.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(history, file)
    print(f"Data saved to existing file: {file_path}")


file_path = r'/storage/brno2/home/yauheni/pneumania_illness/history_val_pneumania.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(history1, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(history1, file)
    print(f"Data saved to existing file: {file_path}")


file_path = r'/storage/brno2/home/yauheni/pneumania_illness/acc_train_pneumania.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_train, file)
    print(f"Data saved to existing file: {file_path}")



file_path = r'/storage/brno2/home/yauheni/pneumania_illness/acc_val_pneumania.pkl'

# Проверка существования файла и запись данных
if not os.path.exists(file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"File created and data saved: {file_path}")
else:
    with open(file_path, 'wb') as file:
        pickle.dump(acc_val, file)
    print(f"Data saved to existing file: {file_path}")



print("Accurancy", correct/total)
print("Train history", history)
print("Validation hisoty", history1)
print("Accuracy train", acc_train)
print("Accuracy validation", acc_val)


print("The end")





    





