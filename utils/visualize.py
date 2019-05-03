import torch
from torchvision import transforms, utils

from .constants import device

def unormalize(tensor):
    invTrans = transforms.Compose(
        [transforms.Normalize(mean=[0., 0., 0.], std=[1/0.5, 1/0.5, 1/0.5]),
        transforms.Normalize(mean=[-0.5, -0.5, -0.5], std=[1., 1., 1.],)
    ])
    return invTrans(tensor)


def get_grid(arr, dataset, network=None):
    grid = torch.zeros(len(arr), 3, 224, 160).to(device)
    for grid_idx, i in enumerate(arr):
        grid[grid_idx], _, _, _ = dataset[i]

    if network:
        network.eval()
        grid, _, _, _ = network(grid, grid)

    grid = utils.make_grid(grid, nrow=4)
    grid = unormalize(grid)
    grid = grid.permute(1, 2, 0).detach().cpu().numpy()

    return grid

def mAP(vectors, t):
    special_frames = list(range(0, len(vectors), 60))
    results = []
    normalized_vectors = vectors / torch.norm(vectors, 2, 1).view(-1, 1)
    for special_frame in special_frames:
        y_true = torch.tensor([abs(i-special_frame)<=t for i in range(len(vectors))])
        target = normalized_vectors[special_frame].view(-1, 1)
        scores = torch.mm(normalized_vectors, target)
        y_true[special_frame] = 0
        scores[special_frame] = min(scores).item()
        results.append(average_precision_score(y_true, scores))
    print(np.mean(results))
