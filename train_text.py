import os
import torch
import torch.nn.functional as F
import argparse
import torch.optim as optim
from time import perf_counter
from torch.utils.data import DataLoader

from utils.constants import *
from utils.dataset import TextTrainingDataset, my_collate
from networks.models import Autoencoder, TextModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--save_frequency', default=1, type=int)
    args = parser.parse_args()

    print('Loading Autoencoder')
    autoencoder = Autoencoder().to(device)
    #autoencoder.load_state_dict(torch.load('weights/model1.pth'))
    autoencoder.eval()

    print('Creating Model')
    text_model = TextModel().to(device)
    optimizer = optim.Adam(text_model.parameters(), lr=args.lr)

    print('Loading Data')
    dataset = TextTrainingDataset('../mario')
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=True, collate_fn=my_collate)

    for epoch in range(1, args.epochs+1):
        length = len(loader)
        t0 = perf_counter()
        for i, (img1, img2, sent, target) in enumerate(loader):
            with torch.no_grad():
                embedding1 = autoencoder.flatten_forward(img1)
                embedding2 = autoencoder.flatten_forward(img2)

            optimizer.zero_grad()
            pred = text_model(sent, embedding1, embedding2)
            loss = F.binary_cross_entropy(F.sigmoid(pred), target)
            loss.backward()
            optimizer.step()

            print('Train Epoch: {} [{}/{} ({:.1f}%)]\tLoss: {:.4f} - {}s'.format(
                epoch, i + 1, length, 100. * (i + 1) / length, loss.item(),
                int(perf_counter() - t0)), end='\r')

            torch.cuda.empty_cache()

        if epoch % args.save_frequency == 0:
            torch.save(text_model.state_dict(), 'weights/text_model{}.pth'.format(epoch))
