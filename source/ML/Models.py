import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal


class LSTMSimple(nn.Module):

    def __init__(self, input_dim, hidden_dim=100, layer_dim=1, output_dim=1):
        """
        Simple LSTM network, which can only predict 1 value
        Input is given to a LSTM, output of the LSTM is the input for a LinearLayer
        Output of the LinearLayer is the prediction
        """
        super(LSTMSimple, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (x, _) = self.lstm(x)  # x.shape = [lstm_layer, batch, hidden_dim]
        x = x[-1]  # last lstm_layer; x.shape = [batch, hidden_dim]
        x = self.linear(x)  # x.shape = [batch, output_dim]
        return x


class LSTMSeq2SeqOle(nn.Module):
    def __init__(self, input_dim, out_seq_len=1, hidden_dim=100, layer_dim=1, output_dim=1):
        """
        LSTM network to predict multiple values
        Uses one LSTM to create a vector-representation of the input sequence, then creates a new sequence with
        that vector as feature for every timestep as input for a second LSTM;
        Output of the second LSTM at each timestep is given to a LinearLayer
        Output of the LinearLayer is the prediction
        :param input_dim: Number of features for each timestep in the sequence used as the input
        :param out_seq_len: Number of predicions
        :param hidden_dim:
        :param layer_dim:
        :param output_dim: Number of features for each prediction
        """
        super(LSTMSeq2SeqOle, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.out_seq_len = out_seq_len
        self.lstm_in = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm_repeat = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Input is encoded by lstm_in
        Output of lstm_in is the input of lstm_repeat; lstm_repeat starts with new empty states (h,c)
        Output of lstm_in is repeated out_seq_len times to get out_seq_len long sequences
        """
        # encode
        _, (x, _) = self.lstm_in(x)  # x.shape = [lstm_layer, batch, hidden_dim]
        x = x[-1]  # last lstm_layer; x.shape = [batch, hidden_dim]
        x = x.unsqueeze(1).repeat(1, self.out_seq_len, 1)  # x.shape = [batch, out_seq_len, hidden_dim]

        # decode
        x, (_, _) = self.lstm_repeat(x)  # x.shape = [batch, out_seq_len, hidden_dim]
        x = self.linear(x)  # x.shape = [batch, out_seq_len, output_dim]
        if x.shape[-1] == 1:
            x = x.view(x.shape[0], -1)  # if output_dim = 1, merge last two dimensions
        return x


class LSTMSeq2SeqBook(nn.Module):
    def __init__(self, input_dim, out_seq_len=1, hidden_dim=100, layer_dim=1, output_dim=1):
        """
        LSTM network to predict multiple values
        Uses one LSTM to process the input and uses the final cell/hidden-state as seed for a second LSTM,
        then creates a new sequence with zero-vector as feature for every timestep as input for the second LSTM;
        Output of the second LSTM at each timestep is given to a LinearLayer
        Output of the LinearLayer is the prediction
        :param input_dim: Number of features for each timestep in the sequence used as the input
        :param out_seq_len: Number of predicions
        :param hidden_dim:
        :param layer_dim:
        :param output_dim: Number of features for each prediction
        """
        super(LSTMSeq2SeqBook, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.out_seq_len = out_seq_len
        self.lstm_in = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm_repeat = nn.LSTM(1, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.register_buffer("device_handler", torch.zeros(1))

    def forward(self, x):
        """
        Input is encoded by lstm_in
        Output of lstm_in is the starting state for lstm_repeat (h, c)
        Input for lstm_repeat is zero ([batch, out_seq_len, 1])
        """
        # encode
        _, (h, c) = self.lstm_in(x)
        # decode
        x, (_, _) = self.lstm_repeat(self.device_handler.new_zeros(x.shape[0], self.out_seq_len, 1), (h, c))
        # x.shape = [batch, out_seq_len, hidden_dim]
        x = self.linear(x)  # x.shape = [batch, out_seq_len, output_dim]
        if x.shape[-1] == 1:
            x = x.view(x.shape[0], -1)  # if output_dim = 1, merge last two dimensions
        return x


class LSTMSeq2SeqBookTranslation(nn.Module):
    def __init__(self, input_dim, out_seq_len=1, hidden_dim=100, layer_dim=1, output_dim=1):
        """
        LSTM network to predict multiple values
        Uses one LSTM to process the input and uses the final cell/hidden-state as seed for a second LSTM,
        then creates a new sequence where the feature of timestep x is the output of the network at timestep x-1;
        Output of the second LSTM at each timestep is given to a LinearLayer
        Output of the LinearLayer is the prediction
        The forward function requires the correct labels as additional parameter during training
        :param input_dim: Number of features for each timestep in the sequence used as the input
        :param out_seq_len: Number of predicions
        :param hidden_dim:
        :param layer_dim:
        :param output_dim: Number of features for each prediction
        """
        super(LSTMSeq2SeqBookTranslation, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.out_seq_len = out_seq_len
        self.output_dim = output_dim
        self.lstm_in = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.lstm_repeat = nn.LSTM(1, hidden_dim, layer_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.register_buffer("device_handler", torch.zeros(1))

    def forward(self, x, y=None):
        """
        During training y must be the correct labels. After training y must be None
        Input is encoded by lstm_in
        Output of lstm_in is the starting state for lstm_repeat (h, c)
        Input for lstm_repeat during training:
            the labels shifted by one (timestep x gets the correct value of timestep x-1)
            last label is discarded, first label is zero
        Input for lstm_repeat after training:
            Output of lstm_repeat at timestep x-1 is the input for timestep x
            in this case: output is a vector but input must be len 1, thus the output at timestep x-1 is used
            for a linear layer first
        """
        # encode
        _, (h, c) = self.lstm_in(x)
        # decode
        if y is not None:
            decoder_input = y.unsqueeze(-1).roll(1, dims=1)
            decoder_input[:, 0] = 0
            x, (_, _) = self.lstm_repeat(decoder_input, (h, c))
            x = self.linear(x)
        else:
            # seq_len per input must be 1 since we feed the seq manually
            # feature is the output of the last iteration, zero for first
            decoder_input = self.device_handler.new_zeros(x.shape[0], 1, self.output_dim)
            # out_seq_len, batch, output
            x = self.device_handler.new_zeros(self.out_seq_len, x.shape[0], self.output_dim)
            for i in range(self.out_seq_len):
                _, (h, c) = self.lstm_repeat(decoder_input, (h, c))
                x[i] = self.linear(h[-1])  # h[-1] = last layer
                decoder_input = x[i].unsqueeze(1)  # unsqueeze to add seq_len 1
            # x.shape = [out_seq_len, batch, self.output_dim]
            x = x.transpose(0, 1)  # batch first
        # x.shape = [batch, out_seq_len, output_dim]
        if x.shape[-1] == 1:
            x = x.view(x.shape[0], -1)  # if output_dim = 1, merge last two dimensions
        return x


class LSTMSeq2SeqDistribution(nn.Module):
    def __init__(self, input_dim, out_seq_len=1, hidden_dim=100, layer_dim=1, output_dim=1):
        """
        LSTM network to predict multiple values
        Uses one LSTM to learn a normal distribution, then creates a new sequence with samples from the distribution
        as feature for every timestep as input for the second LSTM;
        Output of the second LSTM at each timestep is given to a LinearLayer
        Output of the LinearLayer is the prediction
        :param input_dim: Number of features for each timestep in the sequence used as the input
        :param out_seq_len: Number of predicions
        :param hidden_dim:
        :param layer_dim:
        :param output_dim: Number of features for each prediction
        """
        super(LSTMSeq2SeqDistribution, self).__init__()
        self.hidden_dim = hidden_dim
        self.layer_dim = layer_dim
        self.out_seq_len = out_seq_len
        self.lstm_in = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear_mu = nn.Linear(hidden_dim, hidden_dim)
        self.linear_log_std = nn.Linear(hidden_dim, hidden_dim)
        self.lstm_repeat = nn.LSTM(hidden_dim, hidden_dim, layer_dim, batch_first=True)
        self.linear_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Input is encoded by lstm_in
        Output of lstm_in is the input of lstm_repeat; lstm_repeat starts with new empty states (h,c)
        Output of lstm_in is repeated out_seq_len times to get out_seq_len long sequences
        """
        # https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
        # encode
        _, (x, _) = self.lstm_in(x)  # x.shape = [lstm_layer, batch, hidden_dim]
        x = x[-1]  # last lstm_layer; x.shape = [batch, hidden_dim]
        mu = self.linear_mu(x)  # mu.shape = [batch, hidden_dim]
        log_std = self.linear_log_std(x)  # log_std.shape = [batch, hidden_dim]
        dist = Normal(mu, torch.exp(torch.clamp(log_std, -8, 8)))
        x = dist.rsample()  # x.shape = [batch, hidden_dim]
        x = x.unsqueeze(1).repeat(1, self.out_seq_len, 1)  # x.shape = [batch, out_seq_len, hiddem_dim]
        # decode
        x, (_, _) = self.lstm_repeat(x)  # x.shape = [batch, out_seq_len, hidden_dim]
        x = self.linear_output(x)  # x.shape = [batch, out_seq_len, output_dim]
        if x.shape[-1] == 1:
            x = x.view(x.shape[0], -1)  # if output_dim = 1, merge last two dimensions
        return x
