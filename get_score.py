import torch
import argparse
import numpy as np
from time import perf_counter
from torch.utils.data import DataLoader

from utils.dataset import AETestingDataset
from networks.models import Autoencoder
from utils.constants import device


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', default=8, type=int)
    args = parser.parse_args()

    #Split data
    print('Loading Data', end='\r')
    dataset = AETestingDataset('../data/mario/')
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    #Create model
    print('Creating Model', end='\r')
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load('weights/model1.pth'))


    #Start training
    model.eval()
    length = len(dataloader)
    t0 = perf_counter()

    embeddings = []
    embedding_names = []

    for i, (img, name) in enumerate(dataloader):
        if i > 10: break
        embedding = net.flatten_forward(img)
        embeddings.extend(embedding.cpu().detach().numpy())
        embedding_names.extend(name)

        print('{}/{} ({:.1f}%) - {}s'.format(
            i + 1, length, 100. * (i + 1) / length,
            int(perf_counter() - t0)), end='\r')

        del img, name, embedding
        torch.cuda.empty_cache()

    np.save('embedding.npy', embeddings)
    np.save('embedding_names.npy', embedding_names)
