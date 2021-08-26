
from pathlib import Path
import numpy as np

import torch
from torch.utils.data import DataLoader
from skimage.io import imread
from skimage.transform import resize

from mlresearch.models.segnet import SegNet
from mlresearch.loss import bce_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

root = 'datasets/PH2Dataset/PH2 Dataset images'
images = [imread(path) for path in Path(root).glob('**/*Dermoscopic_Image/*.bmp')] # noqa
lesions = [imread(path) for path in Path(root).glob('**/*lesion.bmp')]  # noqa
print('images ', len(images))
print('lessons', len(lesions))

size = (256, 256)
X = [resize(x, size, mode='constant', anti_aliasing=True) for x in images]
Y = [resize(y, size, mode='constant', anti_aliasing=False) > 0.5 for y in lesions]

X = np.array(X, np.float32)
Y = np.array(Y, np.float32)
print(f'Loaded {len(X)} images')


ix = np.random.choice(len(X), len(X), False)
tr, val, ts = np.split(ix, [100, 150])

batch_size = 25

data_tr = DataLoader(list(zip(np.rollaxis(X[tr], 3, 1), Y[tr, np.newaxis])), batch_size=batch_size, shuffle=True)
data_val = DataLoader(list(zip(np.rollaxis(X[val], 3, 1), Y[val, np.newaxis])), batch_size=batch_size, shuffle=True)
data_ts = DataLoader(list(zip(np.rollaxis(X[ts], 3, 1), Y[ts, np.newaxis])), batch_size=batch_size, shuffle=True)


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
            Y_pred = model.forward(X_batch)
            loss = loss_fn(Y_pred, Y_batch)  # forward-pass
            loss.backward()  # backward-pass
            opt.step()  # update weights

            avg_loss += loss / len(data_tr)

        print(f'Epoch {epoch+1}/{epochs}, loss: {avg_loss}')


model = SegNet().to(device)
print(model)

max_epochs = 20
optimizer = torch.optim.Adam(model.parameters())
train(model, optimizer, bce_loss, max_epochs, data_tr, data_val)