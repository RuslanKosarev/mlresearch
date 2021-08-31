
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize

from mlresearch.config import load_config
from mlresearch.models.segnet import SegNet
from mlresearch import loss


config = load_config()
print(config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

path = Path(config.root).expanduser() / config.dataset.path
images = [imread(file) for file in path.glob('**/*Dermoscopic_Image/*.bmp')]
lesions = [imread(file) for file in path.glob('**/*lesion.bmp')]
print('images ', len(images))
print('lessons', len(lesions))

X = [resize(x, config.image.size, mode='constant', anti_aliasing=True) for x in images]
Y = [resize(y, config.image.size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')


ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [100, 150])

data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])), batch_size=config.batch_size, shuffle=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])), batch_size=config.batch_size, shuffle=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])), batch_size=config.batch_size, shuffle=True)


model = SegNet().to(device)
print(model)


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


# def predict(model, data):
#     model.eval()  # testing mode
#     Y_pred = [model(X_batch) for X_batch, _ in data]
#     return np.array(Y_pred)


def score_model(model, loss_fn, metric, data):
    model.eval()
    loss = 0
    score = 0

    for x_batch, y_batch in data:
        x_batch = model(x_batch.to(device))
        y_batch = y_batch.to(device)

        loss += loss_fn(x_batch, y_batch).item()
        score += metric(torch.round(torch.sigmoid(x_batch)).detach(), y_batch).mean()

    loss /= len(data)
    score /= len(data)

    return loss, score


def train(model, optimizer, loss_fn, epochs, data_tr, data_val):

    for epoch in range(epochs):
        avg_loss = 0
        model.train()  # train mode

        for idx, (x_batch, y_batch) in enumerate(data_tr):
            # data to device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            # set parameter gradients to zero
            optimizer.zero_grad()

            # forward
            loss = loss_fn(model(x_batch), y_batch)  # forward-pass
            loss.backward()  # backward-pass
            optimizer.step()  # update weights

            avg_loss += loss.item() / len(data_tr)

        val_loss, val_score = score_model(model, loss_fn, iou_pytorch, data_val)
        print(f'epoch {epoch+1}/{epochs}, loss: {avg_loss}: val_loss {val_loss}, val_score {val_score}')


loss = loss.FocalLoss()
optimizer = torch.optim.Adam(model.parameters())
print('loss', loss)
print('optimizer', optimizer)

train(model, optimizer, loss, config.train.max_epochs, data_tr, data_val)
