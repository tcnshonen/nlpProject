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
    parser.add_argument('weight_path', default='weights/model1.pth', type=str)
    args = parser.parse_args()

    #Split data
    print('Loading Data', end='\r')
    dataset = AEDataset('../data/mario/')
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers=0)

    #Create model
    print('Creating Model', end='\r')
    model = Autoencoder().to(device)
    model.load_state_dict(torch.load(args.weight_path))


    #Start training
    model.eval()
    length = len(loader)
    t0 = perf_counter()

    embeddings = []
    embedding_nums = []
    embedding_actions = []

    for i, (img, ram) in enumerate(loader):
        embedding = net.flatten_forward(img)
        embeddings.extend(embedding.cpu().detach().numpy())
        embedding_nums.extend(num)
        embedding_actions.extend(action)

        print('{}/{} ({:.1f}%) - {}s'.format(
            i + 1, length, 100. * (i + 1) / length,
            int(perf_counter() - t0)), end='\r')

        del img, num, action, embedding
        torch.cuda.empty_cache()

    return embeddings, embedding_nums, embedding_actions
