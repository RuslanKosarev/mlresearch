
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize

from mlresearch.config import load_config
from mlresearch.models.segnet import SegNet
from mlresearch.loss import bce_loss


config = load_config({
    'dataset': 'PH2Dataset/PH2 Dataset images',
    'batch_size': 25,
    'size': (256, 256),
})

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

dataset = Path(config.root).expanduser() / config.dataset
images = [imread(file) for file in dataset.glob('**/*Dermoscopic_Image/*.bmp')]
lesions = [imread(file) for file in dataset.glob('**/*lesion.bmp')]
print('images ', len(images))
print('lessons', len(lesions))

X = [resize(x, config.size, mode='constant', anti_aliasing=True) for x in images]
Y = [resize(y, config.size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')


ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [100, 150])

data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])), batch_size=config.batch_size, shuffle=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])), batch_size=config.batch_size, shuffle=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])), batch_size=config.batch_size, shuffle=True)


def train(model, opt, loss_fn, epochs, data_tr, data_val):
    X_val, Y_val = next(iter(data_val))

    for epoch in range(epochs):
        # tic = time()
        avg_loss = 0
        model.train()  # train mode

        for X_batch, Y_batch in data_tr:

            # data to device
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)

            # set parameter gradients to zero
            opt.zero_grad()

            # forward
            Y_pred = model(X_batch)
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            avg_loss += loss / len(data_tr)

        print(f'Epoch {epoch+1}/{epochs}, loss: {avg_loss}')


model = SegNet().to(device)
print(model)

max_epochs = 2
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, bce_loss, max_epochs, data_tr, data_val)


def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # BATCH x 1 x H x W => BATCH x H x W
    labels = labels.squeeze(1).byte()
    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresholds

    return thresholded


def predict(model, data):
    model.eval()  # testing mode
    Y_pred = [model(X_batch) for X_batch, _ in data]
    return np.array(Y_pred)


def score_model(model, metric, data):
    model.eval()
    scores = 0

    for X_batch, Y_label in data:
        # Y_pred = torch.round(torch.sigmoid(model(X_batch))).detach()
        Y_pred = torch.round(torch.nn.Sigmoid()(model(X_batch))).detach()

        scores += metric(Y_pred, Y_label).mean().item()

    return scores/len(data)


out = score_model(model, iou_pytorch, data_val)
print(out)
