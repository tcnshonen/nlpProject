import torch
import torch.nn as nn

from .layers import ConvLayer, LinearLayer
from utils.constants import word_to_ix, ram_size


class TextOffsetModel(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=64, lstm_dim=128):
        super(TextOffsetModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim

        self.word_embeddings = nn.Embedding(len(word_to_ix), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True,
                            num_layers=1, bidirectional=False)

        # self.fc_emb1 = LinearLayer(self.lstm_dim, 256)
        # self.fc_emb2 = LinearLayer(256, 256)
        # self.fc_emb3 = nn.Linear(256, 256)

        self.fc_frame1 = LinearLayer(2*self.input_dim+self.lstm_dim, 256,
                                     activation_name='leakyrelu', dropout=True)
        self.fc_frame2 = LinearLayer(256, 64, activation_name='leakyrelu', dropout=True)
        # self.fc_frame3 = LinearLayer(128, 64, activation_name='leakyrelu')
        # self.fc_frame4 = LinearLayer(64, 32, activation_name='leakyrelu')
        self.fc_frame3 = nn.Linear(64, 2)

    def forward(self, sent, emb1, emb2):
        x = self.word_embeddings(sent)
        _, (x, _) = self.lstm(x)
        x = x.view(-1, self.lstm_dim)

        # x_emb = self.fc_emb1(x)
        # x_emb = self.fc_emb2(x_emb)
        # x_emb = self.fc_emb3(x_emb)

        emb = torch.cat((emb1, emb2), 1)
        x = torch.cat((emb, x), 1)

        x = self.fc_frame1(x)
        x = self.fc_frame2(x)
        x = self.fc_frame3(x)
        # x = self.fc_frame4(x)
        # x = self.fc_frame5(x)
        x = torch.sigmoid(x)

        return x


class MixModel(nn.Module):
    def __init__(self, embedding_dim=64, lstm_dim=512, bottleneck=256,
                 input_shape=(3, 224, 256), dropout=False,
                 activation_name='relu'):
        super(MixModel, self).__init__()
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim
        self.bottleneck = bottleneck
        self.config = {'dropout': dropout, 'activation_name': activation_name}

        self.encoder_config = {'dropout': dropout, 'activation_name': 'leakyrelu'}

        #Encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 64, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(64, 128, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(128, 256, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(256, 512, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(512, 1024, 4, stride=2, padding=1, **self.encoder_config)
        )

        #Flatten
        self.last_shape, self.flatten_num = self.flatten_shape(self.input_shape)
        self.fc1 = LinearLayer(self.flatten_num, self.bottleneck, **self.config)

        #Memory Decoder
        self.fc_mem = LinearLayer(self.bottleneck, ram_size * 4, **self.config)
        self.memory_decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(ram_size),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(64, 256),
        )

        #RNN
        self.embedding = nn.Embedding(len(word_to_ix), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True)
        self.fc_sen1 = LinearLayer(2*self.bottleneck+self.lstm_dim, 256, activation_name='relu')
        self.fc_sen2 = LinearLayer(256, 64, activation_name='relu')
        self.fc_out = nn.Linear(64, 2)


    def flatten_shape(self, shape):
        temp = torch.rand(1, *shape)
        temp = self.encoder(temp)
        temp_shape = temp.size()
        temp_num = temp.data.view(1, -1).size(1)

        return temp_shape[1:], temp_num

    def flatten_forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flatten_num)
        x = self.fc1(x)

        return x

    def get_diff(self, x1, x2):
        x1 = self.flatten_forward(x1)
        x2 = self.flatten_forward(x2)
        x_diff = x2 - x1

        return x1, x2, x_diff

    def decoder_forward(self, x):
        x = self.fc_mem(x)
        x = x.view(-1, ram_size, 4)
        x = self.memory_decoder(x)
        x.transpose_(1, 2)

        return x

    def text_forward(self, sentence, x1, x2):
        x = self.embedding(sentence)
        _, (x, _) = self.lstm(x)
        x = x.view(-1, self.lstm_dim)

        x_ = torch.cat((x1, x2), 1)
        x = torch.cat((x_, x), 1)
        x = self.fc_sen1(x)
        x = self.fc_sen2(x)
        x = torch.sigmoid(self.fc_out(x))

        return x

    def forward(self, x1, x2, sentence):
        #Encoder
        #x1, x2, diff = self.get_diff(x1, x2)
        x1 = self.flatten_forward(x1)
        x2 = self.flatten_forward(x2)

        #Sentence
        cls = self.text_forward(sentence, x1, x2)

        #Decoder
        x1 = self.decoder_forward(x1)
        x2 = self.decoder_forward(x2)

        return x1, x2, cls


class Autoencoder(nn.Module):
    def __init__(self, input_shape=(3, 224, 256), dropout=False,
                 activation_name='relu'):
        super(Autoencoder, self).__init__()
        self.input_shape = input_shape
        self.config = {'dropout': dropout, 'activation_name': activation_name}

        self.encoder_config = {'dropout': dropout, 'activation_name': 'leakyrelu'}

        #Encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 64, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(64, 128, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(128, 256, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(256, 512, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(512, 1024, 4, stride=2, padding=1, **self.encoder_config)
        )

        #Flatten
        self.last_shape, self.flatten_num = self.flatten_shape(self.input_shape)
        self.fc1 = LinearLayer(self.flatten_num, 256, **self.config)

        #Memory Decoder
        self.fc_mem = LinearLayer(256, ram_size * 4, **self.config)
        self.memory_decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(ram_size),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(64, 256),
        )

    def flatten_shape(self, shape):
        temp = torch.rand(1, *shape)
        temp = self.encoder(temp)
        temp_shape = temp.size()
        temp_num = temp.data.view(1, -1).size(1)
        return temp_shape[1:], temp_num

    def flatten_forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flatten_num)
        x = self.fc1(x)
        return x

    def forward(self, x):
        x = self.flatten_forward(x)

        #Decoder
        x = self.fc_mem(x)
        x = x.view(-1, ram_size, 4)
        x = self.memory_decoder(x)
        x.transpose_(1, 2)

        return x


class TextModel(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=256, lstm_dim=512):
        super(TextModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lstm_dim = lstm_dim

        self.word_embeddings = nn.Embedding(len(word_to_ix), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True,
                            num_layers=1, bidirectional=False)

        self.fc1 = LinearLayer(self.input_dim*2+self.lstm_dim, 512)
        self.fc2 = LinearLayer(512, 256)
        self.fc3 = LinearLayer(256, 128)
        self.fc_out = nn.Linear(128, 2)

    def forward(self, sent, emb1, emb2):
        x = self.word_embeddings(sent)
        _, (x, _) = self.lstm(x)
        x = x.view(-1, self.lstm_dim)

        emb = torch.cat((emb1, emb2), 1)
        x = torch.cat((emb, x), 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc_out(x)

        return x


###############################################################################
###############################################################################
###############################################################################

class OldTextModel(nn.Module):
    def __init__(self, input_dim=256, embedding_dim=256, output_dim=256,
                 lstm_dim=256):
        super(OldTextModel, self).__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.lstm_dim = lstm_dim

        self.word_embeddings = nn.Embedding(len(word_to_ix), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.lstm_dim, batch_first=True)
        self.fc_sen = nn.Linear(self.lstm_dim+self.input_dim, self.output_dim)

    def forward(self, x, y):
        x = self.word_embeddings(x)
        _, (x, _) = self.lstm(x)
        x = x.view(-1, self.lstm_dim)

        x = torch.cat((x, y), 1)
        x = self.fc_sen(x)

        return x


class OldAutoencoder(nn.Module):
    def __init__(self, embedding_dim=256, sentence_num=10, lstm_dim=128,
                 input_shape=(3, 224, 160), dropout=False, activation_name='relu'):
        super(OldAutoencoder, self).__init__()
        self.sentence_num = sentence_num
        self.input_shape = input_shape
        self.lstm_dim = lstm_dim
        self.config = {'dropout': dropout, 'activation_name': activation_name}

        self.encoder_config = {'dropout': dropout, 'activation_name': 'leakyrelu'}

        #Encoder
        self.encoder = nn.Sequential(
            ConvLayer(3, 64, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(64, 128, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(128, 256, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(256, 512, 4, stride=2, padding=1, **self.encoder_config),
            ConvLayer(512, 1024, 4, stride=2, padding=1, **self.encoder_config)
        )

        #Flatten
        self.last_shape, self.flatten_num = self.flatten_shape(self.input_shape)
        self.fc1 = LinearLayer(self.flatten_num, 256, **self.config)

        #Memory Decoder
        self.fc_mem = LinearLayer(256, ram_size * 4, **self.config)
        self.memory_decoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.BatchNorm1d(ram_size),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(64, 256),
        )

        #Image Decoder
        self.fc_img = LinearLayer(256, self.flatten_num, **self.config)
        self.image_decoder = nn.Sequential(
            ConvLayer(1024, 512, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(512, 256, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(256, 128, 4, stride=2, padding=1, transpose=True, **self.config),
            ConvLayer(128, 64, 4, stride=2, padding=1, transpose=True, **self.config),
            nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

        #RNN
        self.word_embeddings = nn.Embedding(len(word_to_ix)-1, 64)
        self.lstm = nn.LSTM(64, self.lstm_dim, batch_first=True)
        self.fc_sen = nn.Linear(self.lstm_dim, 256)


    def flatten_shape(self, shape):
        temp = torch.rand(1, *shape)
        temp = self.encoder(temp)
        temp_shape = temp.size()
        temp_num = temp.data.view(1, -1).size(1)

        return temp_shape[1:], temp_num

    def flatten_forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flatten_num)
        x = self.fc1(x)

        return x

    def get_diff(self, x1, x2):
        x1 = self.flatten_forward(x1)
        x2 = self.flatten_forward(x2)
        x_diff = x2 - x1

        return x1, x2, x_diff

    def decoder_forward(self, x):
        img_x = self.fc_img(x)
        img_x = img_x.view(-1, *self.last_shape)
        img_x = self.image_decoder(img_x)

        mem_x = self.fc_mem(x)
        mem_x = mem_x.view(-1, ram_size, 4)
        mem_x = self.memory_decoder(mem_x)
        mem_x.transpose_(1, 2)

        return img_x, mem_x

    def text_forward(self, sentence):
        x = self.word_embeddings(sentence)
        _, (x, _) = self.lstm(x)
        x = x.view(-1, self.lstm_dim)
        x = self.fc_sen(x)
        return x

    def forward(self, x1, x2, sentence):
        #Encoder
        x1, x2, x_diff = self.get_diff(x1, x2)

        #Decoder
        img_x1, mem_x1 = self.decoder_forward(x1)
        img_x2, mem_x2 = self.decoder_forward(x2)

        #Sentence
        txt_embedding = self.text_forward(sentence)

        return img_x1, mem_x1, img_x2, mem_x2, x_diff, txt_embedding
