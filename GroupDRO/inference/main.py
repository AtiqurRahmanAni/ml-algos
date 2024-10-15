from cub_dataset import CUBDataset
import pandas as pd
import torch
import os
from torch import nn
from utils import *
from loss import LossComputer
from tqdm import tqdm
import sys
import numpy as np
os.environ['HSA_OVERRIDE_GFX_VERSION'] = '10.3.0'
os.environ['HIP_VISIBLE_DEVICES'] = '0'
os.environ['TORCH_HOME'] = '/media/atiqur/New Volume/Programming/ML-Algos/GroupDRO/inference'


batch_size = 64
n_epochs = 10
seed = 0
model = "resnet50"
log_dir = "/media/atiqur/New Volume/Programming/ML-Algos/GroupDRO/inference/test_log"
mode = 'w'

set_seed(seed)

if not os.path.exists(log_dir):
        os.makedirs(log_dir)
logger = Logger(os.path.join(log_dir, 'log.txt'), mode=mode)

data_dir = '/media/atiqur/New Volume/Programming/ML-Algos/datasets/waterbird'
df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))
test_df = df[df['split'] == 2].reset_index(inplace=False)

test_dataset = CUBDataset(df=test_df, data_dir=data_dir)
test_loader = test_dataset.get_loader(batch_size=batch_size, shuffle=False, pin_memory=True)

dataset = {}
dataset['test_data'] = test_dataset
dataset['test_loader'] = test_loader

model = torch.load('/media/atiqur/New Volume/Programming/ML-Algos/GroupDRO/trained_logs/best_model.pth')

logger.flush()

test_csv_logger = CSVBatchLogger(os.path.join(
        log_dir, 'test.csv'), test_dataset.n_groups, mode=mode)


criterion = nn.CrossEntropyLoss(reduction='none')


logger.write(f'\nTest:\n')
loss_computer = LossComputer(
    criterion=criterion,
    dataset=dataset['test_data']
    )

device = get_device()
model = model.to(device)
model.eval()

prog_bar_loader = tqdm(dataset['test_loader'])

results = []

with torch.no_grad():
    for batch_idx, batch in enumerate(prog_bar_loader):
        batch = tuple(t.to(device) for t in batch)
        x = batch[0]
        y = batch[1]
        g = batch[2]
        places = batch[3]

        outputs = model(x)
        preds = outputs.max(dim=1).indices.cpu().numpy()

        for pred, actual, place in zip(preds, y, places):
             results.append({
                  'img_label': actual.item(),
                  'predicted': pred.item(),
                  'place': place.item()
             })

        loss_main = loss_computer.loss(outputs, y, g)

    if loss_computer.batch_count > 0:
        test_csv_logger.log(1, batch_idx, loss_computer.get_stats())
        test_csv_logger.flush()
        loss_computer.log_stats(logger)

logger.write('\n')
test_csv_logger.close()

np.save('/media/atiqur/New Volume/Programming/ML-Algos/GroupDRO/inference/resnet50_inference.npy', results)