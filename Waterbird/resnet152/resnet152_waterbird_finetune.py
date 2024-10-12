import torch
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data import Dataset, DataLoader
from torch import nn


random.seed(42)
np.random.seed(42)
torch.manual_seed(0)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DEVICE


def train(sample: str):
    root_dir = "../../datasets/waterbird_augmented/resnet152_waterbird_features/"

    class WaterbirdDatasetFeatures(Dataset):
        def __init__(self, dataset_path):
            self.dataset = np.load(dataset_path, allow_pickle=True).item()

        def __len__(self):
            return len(self.dataset['features'])

        def __getitem__(self, idx):
            feature = self.dataset['features'][idx]
            label = self.dataset['labels'][idx]

            return feature, label

    batch_size = 64

    saving_dir = "./trained/lr_00001_tanh"

    train_filename = f'train_data_{sample}.npy'
    train_set = WaterbirdDatasetFeatures(
        dataset_path=os.path.join(root_dir, train_filename))
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, pin_memory=True)

    val_filename = 'val_data.npy'
    val_set = WaterbirdDatasetFeatures(
        dataset_path=os.path.join(root_dir, val_filename))
    val_loader = DataLoader(val_set, batch_size=batch_size,
                            shuffle=False, pin_memory=True)

    class MLP(nn.Module):
        def __init__(self):
            super(MLP, self).__init__()
            self.linear = nn.Sequential(nn.Linear(in_features=2048, out_features=1024, bias=False),
                                        nn.BatchNorm1d(num_features=1024),
                                        nn.Tanh(),
                                        nn.Dropout(0.5, inplace=False),
                                        nn.Linear(in_features=1024,
                                                  out_features=512, bias=False),
                                        nn.BatchNorm1d(num_features=512),
                                        nn.Tanh(),
                                        nn.Dropout(0.5, inplace=False),
                                        nn.Linear(in_features=512,
                                                  out_features=1, bias=True),
                                        nn.Sigmoid())

        def forward(self, x):
            out = self.linear(x)
            return out

    model = MLP().to(DEVICE)

    learning_rate = 0.00001
    num_epoch = 20
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)

    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    min_loss = float('inf')

    for epoch in range(num_epoch):

        total_loss = 0
        corrects = 0

        # training loop
        for features, labels in tqdm(train_loader, desc=f'Train epoch: {epoch + 1}'):
            features = features.to(DEVICE)
            labels = labels.view(-1, 1).to(torch.float32).to(DEVICE)

            out = model(features)
            loss = criterion(out, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds = (out >= 0.5).to(torch.float32)
            corrects += torch.sum((preds == labels)).item()

        accuracy = corrects / len(train_loader.dataset)
        epoch_loss = total_loss / len(train_loader)
        train_loss.append(epoch_loss)
        train_acc.append(accuracy)
        # print(
        #     f'Epoch: {epoch + 1} | Loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}')

        # validation loop
        total_loss = 0
        corrects = 0
        with torch.no_grad():
            for features, labels in tqdm(val_loader, desc=f'Validation epoch: {epoch + 1}'):

                features = features.to(DEVICE)
                labels = labels.view(-1, 1).to(torch.float32).to(DEVICE)

                out = model(features)

                loss = criterion(out, labels)
                total_loss += loss.item()

                preds = (out >= 0.5).to(torch.float32)
                corrects += torch.sum((preds == labels)).item()

            accuracy = corrects / len(val_loader.dataset)
            epoch_loss = total_loss / len(val_loader)
            val_loss.append(epoch_loss)
            val_acc.append(accuracy)
            # print(
            #     f'Validation loss: {epoch_loss:.4f} | Accuracy: {accuracy:.2f}')

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                torch.save(model.state_dict(), os.path.join(
                    saving_dir, f"resnet150_augmented_{sample}.pth"))

    train_statistics = {
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_acc': train_acc,
        'val_acc': val_acc
    }

    np.save(os.path.join(
        saving_dir, f'train_statistics_{sample}.npy'), train_statistics)


if __name__ == "__main__":
    for sample_size in [100, 200, 400, 800]:
        for j in range(1, 4):
            sample = f'{sample_size}_sample_{j}'
            print(f'Training for: {sample}')
            train(sample=sample)
            print("\n")
