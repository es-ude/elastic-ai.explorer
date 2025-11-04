from torch import nn


# We don't use the cell states that are returned by the pytorch module. To make it easier to connect the layers,
# this class throws the cell states away and only returns the output.
class SimpleLSTM(nn.Module):
    def __init__(
        self, input_size, hidden_size, num_layers, bidirectional, batch_first, dropout
    ):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            bidirectional=bidirectional,
            batch_first=batch_first,
            dropout=dropout,
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out
