import torch
from torchvision import transforms

ram_size = 4096

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


sentence_dic = {
    'left_move': 'go to left',
    'left_jump': 'go to left',
    'left_spin': 'go to left',
    'right_move': 'go to right',
    'right_jump': 'go to right',
    'right_spin': 'go to right',
}

sentence_dic_ = {
    'left_move': 'move toward left',
    'left_jump': 'jump toward left',
    'left_spin': 'spin toward left',
    'right_move': 'move toward right',
    'right_jump': 'jump toward right',
    'right_spin': 'spin toward right',
}

word_to_ix = {'': 0}
for s in sentence_dic.values():
    for word in s.split():
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
