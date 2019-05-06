import torch
from torchvision import transforms

ram_size = 4096

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

action_list = ['left_move', 'left_jump', 'left_spin',
               'right_move', 'right_jump', 'right_spin']

sentence_dic = {
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

sentence_dic_ = {
    'climb_down_ladder': 'climb down the ladder',
    'right_jump_off_rope': 'jump off of the rope to the right',
    'right_jump_over_monster': 'jump over the monster to the right',
    'right_jump_over_wood': 'jump over the wood to the right',
    'right_jump_to_rope': 'jump to the rope on the right',
    'left_jump_off_rope': 'jump off of the rope to the left',
    'right_move': 'move toward the right',
    'swing_to_right': 'swing to the right',
    'right_jump': 'jump to the right',
    'fall_down_pit': 'fall down into the pit',
    'left_jump_over_pit': 'jump over the pit to the left',
    'jump_up': 'jump up',
    'jump_down': 'jump down',
    'left_jump_over_monster': 'jump over the monster to the left',
    'left_jump_to_rope':  'jump to the rope on the left',
    'climb_up_ladder': 'climb up the ladder',
    'left_move': 'move toward the left',
    'left_jump': 'jump to the left',
    'right_jump_over_fire': 'jump over the fire to the right',
    'swing_to_left': 'swing to the left',
    'right_jump_over_pit': 'jump over the pit to the right',
    'left_jump_over_fire': 'jump over the fire to the left',
    'left_jump_over_wood': 'jump over the wood to the left'
}
