import torch
import torch.nn as nn
from pytorch_lightning import LightningModule, Trainer


class FrequencyEmbedding(nn.Module):
    def __init__(self, d_model):
        """
        :param d_model: int, output dimension of the embedding layer
        """

        super().__init__()
        self.freq_embedding = nn.Linear(1, d_model)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, 1)
        :return: (batch_size, seq_len, d_model)
        """
        output = self.freq_embedding(x)
        return output


class OperatorEmbedding(nn.Module):
    def __init__(self, in_dim, d_model):
        """
        :param in_dim: int, dimension of the row (col) of the operator matrix
        :param d_model: int, output dimension of the embedding layer
        """

        super().__init__()
        self.operator_embedding = nn.Linear(2*in_dim**2, d_model)

    def forward(self, x):
        """
        :param x: (batch_size, seq_len, in_dim, in_dim), complex matrix
        :return: (batch_size, seq_len, d_model)
        """

        # flatten
        x = x.view(x.shape[0], x.shape[1], -1)  # (batch_size, seq_len, in_dim**2)

        # separate real and imaginary part
        x_real = torch.real(x)
        x_imag = torch.imag(x)

        # concatenate
        x = torch.cat((x_real, x_imag), dim=-1)  # (batch_size, seq_len, 2*in_dim**2)

        output = self.operator_embedding(x)
        return output


class InputEmbedding(nn.Module):
    def __init__(self, operator_dim, d_model):
        """
        :param operator_dim: int, dimension of the row (col) of the operator matrix
        :param d_model: int, output dimension of the embedding layer
        """

        super().__init__()
        self.freq_embedding = FrequencyEmbedding(d_model)
        self.operator_embedding = OperatorEmbedding(operator_dim, d_model)

        self.mix = nn.Linear(2*d_model, d_model)

    def forward(self, freq, operator):
        """
        :param freq: (batch_size, seq_len, 1)
        :param operator: (batch_size, seq_len, operator_dim, operator_dim), complex matrix
        :return: (batch_size, seq_len, d_model)
        """

        freq_emb = self.freq_embedding(freq)
        operator_emb = self.operator_embedding(operator)

        output = self.mix(torch.cat((freq_emb, operator_emb), dim=-1))
        return output


class StateEncoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layers, activation="relu"):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=n_layers
        )

    def forward(self, src):
        """
        :param src: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        output = self.transformer_encoder(src)
        return output


class StateDecoder(nn.Module):
    def __init__(self, out_dim, d_model, nhead, dim_feedforward, dropout, n_layers, activation="relu"):
        super().__init__()
        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )

        self.transformer_decoder = nn.TransformerEncoder(
            self.decoder_layer, num_layers=n_layers
        )

        self.last_layer = nn.Linear(d_model, out_dim**2)
        self.unflatten = nn.Unflatten(-1, (out_dim, out_dim))

    def forward(self, src):
        """
        :param src: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        output = self.transformer_decoder(src)

        output = self.last_layer(output)
        output = self.unflatten(output)  # (batch_size, seq_len, out_dim, out_dim)

        return output


class FrequencyDecoder(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, n_layers, activation="relu", apply_softmax=False):
        super().__init__()

        self.apply_softmax = apply_softmax

        self.decoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, batch_first=True
        )

        self.transformer_decoder = nn.TransformerEncoder(
            self.decoder_layer, num_layers=n_layers
        )

        self.last_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        """
        :param src: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        output = self.transformer_decoder(src)

        output = self.last_layer(output)

        if self.apply_softmax:
            output = torch.softmax(output, dim=-1)

        return output


class SOTA(LightningModule):
    def __init__(self, operator_dim, d_model, nhead, dim_feedforward, dropout, n_layers, activation="relu"):
        super().__init__()

        self.input_embedding = InputEmbedding(operator_dim, d_model)
        self.operator_embedding = OperatorEmbedding(operator_dim, d_model)
        self.state_encoder = StateEncoder(d_model, nhead, dim_feedforward, dropout, n_layers, activation)
        self.state_decoder = StateDecoder(operator_dim, d_model, nhead, dim_feedforward, dropout, n_layers, activation)

        self.state_loss = nn.MSELoss()

    def forward(self, imperfect_freq, imperfect_operator, masked_operator):
        """
        :param freq: (batch_size, seq_len, 1)
        :param operator: (batch_size, seq_len, operator_dim, operator_dim), complex matrix
        :param masked_operator: (batch_size, seq_len1, operator_dim, operator_dim), complex matrix
        :return: (batch_size, seq_len, operator_dim, operator_dim), complex matrix
        """
        input_emb = self.input_embedding(imperfect_freq, imperfect_operator)  # (batch_size, seq_len, d_model)
        masked_operator_emb = self.operator_embedding(masked_operator)  # (batch_size, seq_len1, d_model)

        state_emb = self.state_encoder(input_emb) # (batch_size, seq_len, d_model)

        state_emb = torch.cat((state_emb, masked_operator_emb), dim=1)  # (batch_size, seq_len+seq_len1, d_model)

        state = self.state_decoder(state_emb)

        return state

    def training_step(self, batch, batch_idx):
        """
        :param batch: (imperfect_freq, imperfect_operator, masked_operator, state, proba)
            imperfect_freq: is the set of frequencies, missing m frequencies
            imperfect_operator: is the set of operators, missing m frequencies
            masked_operator: is the set of m missing operators
            state: is the set of states
            proba: is the set of probabilities
        :param batch_idx:
        :return:
        """
        imperfect_freq, imperfect_operator, masked_operator, state, proba = batch

        state_hat = self(imperfect_freq, imperfect_operator, masked_operator)

        loss = self.state_loss(state_hat, state)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        """
        :param batch: (freq, operator, state)
        :param batch_idx:
        :return:
        """
        pass


class QSTDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    pass