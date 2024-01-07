import torch
import torch.nn as nn
import pickle
import tqdm


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

        self.last_layer = nn.Linear(d_model, 2*out_dim**2)
        self.unflatten = nn.Unflatten(-1, (2, out_dim, out_dim))

    def forward(self, src):
        """
        :param src: (batch_size, seq_len, d_model)
        :return: (batch_size, seq_len, d_model)
        """
        output = self.transformer_decoder(src)

        output = torch.mean(output, dim=1)  # (batch_size, d_model)

        output = self.last_layer(output)

        output = self.unflatten(output)  # (batch_size, 2, out_dim, out_dim)

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


class SOTA(nn.Module):
    def __init__(self, operator_dim, d_model, nhead, dim_feedforward, dropout, n_layers, activation="relu"):
        super().__init__()

        self.input_embedding = InputEmbedding(operator_dim, d_model)
        self.operator_embedding = OperatorEmbedding(operator_dim, d_model)
        self.state_encoder = StateEncoder(d_model, nhead, dim_feedforward, dropout, n_layers, activation)
        self.state_decoder = StateDecoder(operator_dim, d_model, nhead, dim_feedforward, dropout, n_layers, activation)

        self.state_loss = nn.MSELoss()

    def forward(self, imperfect_freqs, imperfect_operators, masked_operators):
        """
        :param imperfect_freqs: (batch_size, seq_len, 1)
        :param imperfect_operators: (batch_size, seq_len, operator_dim, operator_dim), complex matrix
        :param masked_operators: (batch_size, seq_len1, operator_dim, operator_dim), complex matrix
        :return: (batch_size, seq_len, operator_dim, operator_dim), complex matrix
        """
        input_emb = self.input_embedding(imperfect_freqs, imperfect_operators)  # (batch_size, seq_len, d_model)
        masked_operators_emb = self.operator_embedding(masked_operators)  # (batch_size, seq_len1, d_model)

        state_emb = self.state_encoder(input_emb) # (batch_size, seq_len, d_model)

        state_emb = torch.cat((state_emb, masked_operators_emb), dim=1)  # (batch_size, seq_len+seq_len1, d_model)

        state = self.state_decoder(state_emb)

        return state


class QSTDataset(torch.utils.data.Dataset):
    def __init__(self):
        with open("dataset.pkl", "rb") as f:
            self.dataset = pickle.load(f)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        freqs, operators, states, probabilities = self.dataset[idx]

        freqs = torch.tensor(freqs, dtype=torch.float32).unsqueeze(-1)
        operators = torch.tensor(operators, dtype=torch.complex64)
        states = torch.tensor(states, dtype=torch.complex64)
        probabilities = torch.tensor(probabilities, dtype=torch.float32)

        # states from (out_dim, out_dim) to (2, out_dim, out_dim) where first dim is real and second is imaginary
        states = torch.stack((torch.real(states), torch.imag(states)), dim=0)

        n = 3  # number of missing frequencies

        imperfect_freqs = freqs[:-n]
        imperfect_operators = operators[:-n]
        masked_operators = operators[-n:]

        return imperfect_freqs, imperfect_operators, masked_operators, states, probabilities


def train():
    dataset = QSTDataset()
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_built()
        else "cpu",)

    model = SOTA(operator_dim=2, d_model=16, nhead=4, dim_feedforward=64, dropout=0, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        pbar = tqdm.tqdm(train_loader)
        for batch in pbar:
            imperfect_freqs, imperfect_operators, masked_operators, state, proba = batch

            imperfect_freqs = imperfect_freqs.to(device)
            imperfect_operators = imperfect_operators.to(device)
            masked_operators = masked_operators.to(device)
            state = state.to(device)
            proba = proba.to(device)

            state_hat = model(imperfect_freqs, imperfect_operators, masked_operators)

            loss = loss_fn(state_hat, state)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(f"Epoch {epoch+1} | Loss {loss.item():.4f}")


if __name__ == "__main__":
    train()
