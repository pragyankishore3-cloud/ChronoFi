from torch import nn


class ChronoFiLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=50, output_size=1, num_layers=2, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        # The LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_layer_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout for Monte Carlo Dropout during inference (to simulate risk/uncertainty)
        self.dropout = nn.Dropout(dropout)

        # The Linear layer that maps the hidden state output to the final prediction
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # input_seq shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(input_seq)
        
        # Take the output from the last time step
        last_time_step_out = lstm_out[:, -1, :]
        
        # Apply dropout (even during inference if we enable train mode for MC Dropout)
        dropped_out = self.dropout(last_time_step_out)
        
        predictions = self.linear(dropped_out)
        return predictions
