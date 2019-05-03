import torch
import argparse
import numpy as np
from time import perf_counter
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from utils.dataset import AETestingDataset
from networks.models import Autoencoder
from utils.constants import device


action_list = ['left_move', 'left_jump', 'left_spin',
               'right_move', 'left_jump', 'left_spin']

def get_separate_embedding(frame):
    frame = str(frame)
    total_embeddings = []
    total_actions = []
    for i, action in enumerate(action_list):
        indices = [j for j, x in enumerate(embedding_names)\
            if x.split('\\')[0] == action and x.split('\\')[2].split('.')[0] == frame]
        total_embeddings.extend(embeddings[indices])
        total_actions.extend([i] * len(indices))

    return np.array(total_embeddings), np.array(total_actions)


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--batch_size', default=8, type=int)
    # args = parser.parse_args()
    #
    # #Split data
    # print('Loading Data', end='\r')
    # dataset = AETestingDataset('../data/mario/')
    # dataloader = DataLoader(dataset, batch_size=args.batch_size,
    #                         shuffle=False, num_workers=0)
    #
    # #Create model
    # print('Creating Model', end='\r')
    # model = Autoencoder().to(device)
    # model.load_state_dict(torch.load('weights/model1.pth'))
    #
    #
    # #Start training
    # model.eval()
    # length = len(dataloader)
    # t0 = perf_counter()
    #
    # embeddings = []
    # embedding_names = []
    #
    # for i, (img, name) in enumerate(dataloader):
    #     if i > 10: break
    #     embedding = model.flatten_forward(img)
    #     embeddings.extend(embedding.cpu().detach().numpy())
    #     embedding_names.extend(name)
    #
    #     print('{}/{} ({:.1f}%) - {}s'.format(
    #         i + 1, length, 100. * (i + 1) / length,
    #         int(perf_counter() - t0)), end='\r')
    #
    #     del img, name, embedding
    #     torch.cuda.empty_cache()
    #
    # np.save('embeddings.npy', embeddings)
    # np.save('embedding_names.npy', embedding_names)

    embeddings = np.load('embeddings.npy')
    embedding_names = np.load('embedding_names.npy')

    # Get total frame
    indices = [i for i, x in enumerate(embedding_names)\
        if x.split('\\')[0] == 'origin']
    origin_embeddings = embeddings[indices]
    origin_embeddings = origin_embeddings / np.linalg.norm(origin_embeddings, axis=1).reshape((-1, 1))


    def get_F1(action):
        action_index = action_list.index(action)
        special_frames = list(range(0, len(origin_embeddings), 60))
        results = []
        for special_frame in special_frames:
            vectors, names = get_separate_embedding(special_frame)
            vectors = vectors / np.linalg.norm(vectors, axis=1).reshape((-1, 1))
            y_true = np.array([name.split('\\') == action_index for name in zip(names)])
            target = origin_embeddings[special_frame].view(-1, 1)
            scores = np.dot(vectors, target)
            results.append(average_precision_score(y_true, scores))
        return np.mean(results)


    # Get F1 score for embeddings
    for action in action_list:
        f1_score = get_F1(action)
        print(action, f1_score)
