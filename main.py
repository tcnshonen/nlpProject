import os
import torch
import argparse
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, random_split

from utils.trainer import train
from utils.constants import device, data_transform
from utils.dataset import ImageStateDataset
from networks.autoencoder import Autoencoder


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--save_frequency', default=5, type=int)
    args = parser.parse_args()

    if not os.path.exists('weights/'):
        os.mkdir('weights/')

    #Split data
    full_dataset = ImageStateDataset('pitfall/')
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(
        full_dataset, [train_size, test_size])
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #Create model
    model = Autoencoder(dropout=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print('Epochs: {}\nBatch Size: {}\nDevice: {}'.format(
        args.epochs, args.batch_size, device))
    print('Total Training Dataset: {}\nTotal Testing Dataset: {}'.format(
        len(train_dataset), len(test_dataset)))


    #Start training
    for epoch in range(1, args.epochs+1):
        train(epoch, model, optimizer, train_dataloader)

        if epoch % args.save_frequency == 0:
            torch.save(model.state_dict(), 'weights/model{}.pth'.format(epoch))
