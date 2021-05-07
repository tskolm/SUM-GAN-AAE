# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from layers.lstmcell import StackedLSTMCell

class FCSN(nn.Module):
    def __init__(self, n_class=2):
        super(FCSN, self).__init__()

        self.conv1 = nn.Sequential(OrderedDict([
            ('conv1_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_1', nn.BatchNorm1d(1024)),
            ('relu1_1', nn.ReLU(inplace=True)),
            ('conv1_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn1_2', nn.BatchNorm1d(1024)),
            ('relu1_2', nn.ReLU(inplace=True)),
            ('pool1', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/2

        self.conv2 = nn.Sequential(OrderedDict([
            ('conv2_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_1', nn.BatchNorm1d(1024)),
            ('relu2_1', nn.ReLU(inplace=True)),
            ('conv2_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn2_2', nn.BatchNorm1d(1024)),
            ('relu2_2', nn.ReLU(inplace=True)),
            ('pool2', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/4

        self.conv3 = nn.Sequential(OrderedDict([
            ('conv3_1', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_1', nn.BatchNorm1d(1024)),
            ('relu3_1', nn.ReLU(inplace=True)),
            ('conv3_2', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_2', nn.BatchNorm1d(1024)),
            ('relu3_2', nn.ReLU(inplace=True)),
            ('conv3_3', nn.Conv1d(1024, 1024, 3, padding=1)),
            ('bn3_3', nn.BatchNorm1d(1024)),
            ('relu3_3', nn.ReLU(inplace=True)),
            ('pool3', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/8

        self.conv4 = nn.Sequential(OrderedDict([
            ('conv4_1', nn.Conv1d(1024, 2048, 3, padding=1)),
            ('bn4_1', nn.BatchNorm1d(2048)),
            ('relu4_1', nn.ReLU(inplace=True)),
            ('conv4_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_2', nn.BatchNorm1d(2048)),
            ('relu4_2', nn.ReLU(inplace=True)),
            ('conv4_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn4_3', nn.BatchNorm1d(2048)),
            ('relu4_3', nn.ReLU(inplace=True)),
            ('pool4', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/16

        self.conv5 = nn.Sequential(OrderedDict([
            ('conv5_1', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_1', nn.BatchNorm1d(2048)),
            ('relu5_1', nn.ReLU(inplace=True)),
            ('conv5_2', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_2', nn.BatchNorm1d(2048)),
            ('relu5_2', nn.ReLU(inplace=True)),
            ('conv5_3', nn.Conv1d(2048, 2048, 3, padding=1)),
            ('bn5_3', nn.BatchNorm1d(2048)),
            ('relu5_3', nn.ReLU(inplace=True)),
            ('pool5', nn.MaxPool1d(2, stride=2, ceil_mode=True))
            ])) # 1/32

        self.conv6 = nn.Sequential(OrderedDict([
            ('fc6', nn.Conv1d(2048, 4096, 1)),
            ('bn6', nn.BatchNorm1d(4096)),
            ('relu6', nn.ReLU(inplace=True)),
            ('drop6', nn.Dropout())
            ]))
   
        self.conv7 = nn.Sequential(OrderedDict([
            ('fc7', nn.Conv1d(4096, 4096, 1)),
            ('bn7', nn.BatchNorm1d(4096)),
            ('relu7', nn.ReLU(inplace=True)),
            ('drop7', nn.Dropout())
            ]))

        self.conv8 = nn.Sequential(OrderedDict([
            ('fc8', nn.Conv1d(4096, n_class, 1)),
            ('bn8', nn.BatchNorm1d(n_class)),
            ('relu8', nn.ReLU(inplace=True)),
            ]))

        self.conv_pool4 = nn.Conv1d(2048, n_class, 1)
        self.bn_pool4 = nn.BatchNorm1d(n_class)

        self.deconv1 = nn.ConvTranspose1d(n_class, n_class, 4, padding=1, stride=2, bias=False)
        self.deconv2 = nn.ConvTranspose1d(n_class, n_class, 16, stride=16, bias=False)
        self.sigm = nn.Sigmoid()

    def forward(self, x):

        h = x
        h = self.conv1(h)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        pool4 = h

        h = self.conv5(h)
        h = self.conv6(h)
        h = self.conv7(h)
        h = self.conv8(h)

        h = self.deconv1(h)
        upscore2 = h

        h = self.conv_pool4(pool4)
        h = self.bn_pool4(h)
        score_pool4 = h

        h = upscore2 + score_pool4

        h = self.sigm(self.deconv2(h))

        return h


class sLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Scoring LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, bidirectional=True)
        self.out = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),  # bidirection => scalar
            nn.Sigmoid())

    def forward(self, features, init_hidden=None):
        """
        Args:
            features: [seq_len, 1, 500] (compressed pool5 features)
        Return:
            scores: [seq_len, 1]
        """
        self.lstm.flatten_parameters()

        # [seq_len, 1, hidden_size * 2]
        features, (h_n, c_n) = self.lstm(features)

        # [seq_len, 1]
        scores = self.out(features.squeeze(1))

        return scores


class eLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        """Encoder LSTM"""
        super().__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)

    def forward(self, frame_features):
        """
        Args:
            frame_features: [seq_len, 1, hidden_size]
        Return:
            output: [seq_len, 1, hidden_size]
            last hidden:
                h_last [num_layers=2, 1, hidden_size]
                c_last [num_layers=2, 1, hidden_size]
        """
        self.lstm.flatten_parameters()
        output, (h_last, c_last) = self.lstm(frame_features)

        return output, (h_last, c_last)


class Attn(nn.Module):
    """Luong attention layer"""
    def __init__(self, hidden_size, n_layers):
        super(Attn, self).__init__()
        self.hidden_size = hidden_size
        self.attn = torch.nn.Linear(self.hidden_size*n_layers, hidden_size)	

    def Mult_score(self, prev_hidden, encoder_output):
        """
        Args:
            prev_hidden: [num_layers, batch_size=1, hidden_size]
            encoder_output: [seq_len=1, batch_size, hidden_size]
        Return:
            [seq_len, batch_size=1]
        """
        prev_hidden = prev_hidden.transpose(0,1)					# [batch_size=1, num_layers, hidden_size]
        prev_hidden = prev_hidden.reshape(prev_hidden.shape[0], -1) # [batch_size=1, num_layers*hidden_size]
        energy = self.attn(prev_hidden)								# [batch_size=1, hidden_size]
        energy = energy.unsqueeze(0) 								# [1, batch_size=1, hidden_size]
        return torch.sum(encoder_output * energy, dim=2)

    def forward(self, prev_hidden, encoder_output):
        """
        Args:
            prev_hidden: [num_layers, batch_size=1, hidden_size]
            encoder_output: [seq_len=1, batch_size, hidden_size]
        Return:
            [batch_size=1, 1, seq_len]
        """
        # Calculate the attention weights (energies)
        attn_energies = self.Mult_score(prev_hidden, encoder_output)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()	# [batch_size=1, seq_len]

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

class dLSTM(nn.Module):
    def __init__(self, input_size=2048, hidden_size=2048, num_layers=2):
        """Decoder LSTM"""
        super().__init__()

        self.attn = Attn(hidden_size, num_layers)
        self.lstm_cell = StackedLSTMCell(num_layers, 2 * input_size, hidden_size)
        self.out = nn.Linear(hidden_size, input_size)

    def forward(self, seq_len, encoder_output, init_hidden):
        """
        Args:
            seq_len: (int)
            encoder_output: [seq_len, 1, hidden_size]
            init_hidden:
                h [num_layers=2, 1, hidden_size]
                c [num_layers=2, 1, hidden_size]
        Return:
            out_features: [seq_len, 1, hidden_size]
        """

        batch_size = init_hidden[0].size(1)
        hidden_size = init_hidden[0].size(2)

        input_step = Variable(torch.zeros(batch_size, hidden_size)).cuda()
        h, c = init_hidden  # (h_0, c_0): last state of eLSTM

        out_features = []
        for i in range(seq_len):
            # last_h: [1, hidden_size] (h from last layer)
            # last_c: [1, hidden_size] (c from last layer)
            # h: [num_layers=2, 1, hidden_size] (h from all layers)
            # c: [num_layers=2, 1, hidden_size] (c from all layers)

            # Calculate attention weights from the previous decoder hidden states
            attn_weights = self.attn(h, encoder_output)  # [1, 1, seq_len]
            # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
            context = attn_weights.bmm(encoder_output.transpose(0, 1))  # [1, 1, hidden_size]

            input_step = input_step.unsqueeze(0)  # [1, 1, hidden_size]
            rnn_input = torch.cat([input_step, context], dim=2)  # [1, 1, 2*hidden_size]
            rnn_input = rnn_input.squeeze(1)

            (last_h, last_c), (h, c) = self.lstm_cell(rnn_input, (h, c))
            input_step = self.out(last_h)
            out_features.append(last_h)
        # list of seq_len '[1, hidden_size]-sized Variables'
        return out_features


class AE(nn.Module):  
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.e_lstm = eLSTM(input_size, hidden_size, num_layers)
        self.d_lstm = dLSTM(input_size, hidden_size, num_layers)

    def forward(self, features):
        """
        Args:
            features: [seq_len, 1, hidden_size]
        Return:
            decoded_features: [seq_len, 1, hidden_size]
        """
        seq_len = features.size(0)

        # encoder_output: [seq_len, 1, hidden_size]
        # h and c: [num_layers, 1, hidden_size]
        encoder_output, (h, c) = self.e_lstm(features)

        # [seq_len, 1, hidden_size]
        decoded_features = self.d_lstm(seq_len, encoder_output, init_hidden=(h, c))

        # [seq_len, 1, hidden_size]
        # reverse
        decoded_features.reverse()
        decoded_features = torch.stack(decoded_features)
        return decoded_features


class Summarizer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.fcsn = FCSN(input_size, hidden_size, num_layers)
        self.auto_enc = AE(input_size, hidden_size, num_layers)

    def forward(self, image_features):
        """
        Args:
            image_features: [seq_len, 1, hidden_size]
        Return:
            scores: [seq_len, 1]
            decoded_features: [seq_len, 1, hidden_size]
        """

        # Apply weights
        # [seq_len, 1]
        scores = self.fcsn(image_features)

        # [seq_len, 1, hidden_size]
        weighted_features = image_features * scores.view(-1, 1, 1)

        decoded_features = self.auto_enc(weighted_features)

        return scores, decoded_features


if __name__ == '__main__':

    pass
