from torch import nn


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional, batch_first):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return last_output


class LinearOne(nn.Module):
    def __init__(self, input, output):
        super(LinearOne, self).__init__()
        self.linear = nn.Linear(input, output)

    def forward(self, x):
        return self.linear(x).squeeze(1)
