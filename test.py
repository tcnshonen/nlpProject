import os
import torch
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.trainer import AE_Trainer
from utils.dataset import AEDataset
from networks.models import Autoencoder
from utils.constants import device, data_transform


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--lr', default=0.003, type=float)
    parser.add_argument('--save_frequency', default=1, type=int)
    args = parser.parse_args()

    if not os.path.exists('weights/'):
        os.mkdir('weights/')

    #Split data
    print('Loading Data', end='\r')
    dataset = AEDataset('../data/mario/')
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

    #Create model
    print('Creating Model', end='\r')
    model = Autoencoder(dropout=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = AE_Trainer(model, optimizer, dataloader)

    print()
    print('Epochs: {}\nBatch Size: {}\nDevice: {}'.format(
        args.epochs, args.batch_size, device))
    print('Total Dataset: {}'.format(len(dataset)))


    #Start training
    for epoch in range(1, args.epochs+1):
        trainer.train(epoch)

        if epoch % args.save_frequency == 0:
            torch.save(model.state_dict(), 'weights/model{}.pth'.format(epoch))
