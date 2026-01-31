"""
Convolutional LSTM Scene-to-Scene Decoder
Revision 01 (first version)
for ADLR Trajectory Planning Project
Moritz Schüler and Alexander João Peterson Santos
2026-01-06
"""

# pytorch imports
from typing import Tuple

# other imports
import numpy as np
import torch
import torch.nn as nn

# constants
PRECISION = np.float32


class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv = nn.Conv2d(
            input_channels + hidden_dim,
            4 * hidden_dim,
            kernel_size,
            padding=padding,
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        conv_out = self.conv(combined)

        i, f, o, g = torch.split(conv_out, self.hidden_dim, dim=1)

        i = torch.sigmoid(i)  # input gate
        f = torch.sigmoid(f)  # forget gate
        o = torch.sigmoid(o)  # output gate
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_dim, kernel_size):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size)
        self.hidden_dim = hidden_dim

    def forward(self, x, hidden_state=None):
        # x: (B, T, C, H, W)

        B, T, C, H, W = x.shape

        # initialize hidden state if not provided
        if hidden_state is None:
            h = torch.zeros(B, self.hidden_dim, H, W, device=x.device)
            c = torch.zeros_like(h)
        else:
            h, c = hidden_state

        outputs = []

        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
            outputs.append(h.unsqueeze(1))

        return torch.cat(outputs, dim=1), (h, c)


class StackedConvLSTM(nn.Module):
    def __init__(
        self, input_channels: int, hidden_dim: int, kernel_size: int, num_layers: int
    ):
        super().__init__()
        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            in_ch = input_channels if i == 0 else hidden_dim
            self.layers.append(ConvLSTM(in_ch, hidden_dim, kernel_size))

    def forward(self, x, hidden_states=None):
        # x: (B, T, C, H, W)
        output = x
        new_states = []

        for i, layer in enumerate(self.layers):
            # get the hidden state for this layer if available
            h_state = hidden_states[i] if hidden_states is not None else None

            output, (h, c) = layer(output, h_state)
            new_states.append((h, c))  # store final hidden + cell states for each layer

        return output, new_states


class LSTMSceneToScene02(nn.Module):
    """
    LSTM-based model for predicting the next frame in a 2D dynamic scene consisting of a sequence of frames.

    Architecture:
        - ConvLSTM to model temporal dependencies
        - Decoder: Conv layer to predict next frame
    """

    frame_dims: Tuple[int, int]
    hidden_dim: int
    num_lstm_layers: int

    def __init__(
        self,
        frame_dims: Tuple[int, int] = (64, 64),
        in_channels: int = 1,
        hidden_dim: int = 64,
        num_lstm_layers: int = 2,
        kernel_size: int = 3,
    ):
        """
        Initialize the LSTM scene-to-scene prediction model.

        Parameters
        ----------
        frame_dims: Tuple[int, int]
            Dimensions of each frame (height, width)
        in_channels: int
            1 input channel for our grayscale input
        hidden_dim: int
            Hidden dimension of the LSTM layers
        num_lstm_layers: int
            Number of stacked LSTM layers
        kernel_size: int
            Size of the kernels in the ConvLSTM layers

        """
        super(LSTMSceneToScene02, self).__init__()

        # set attributes
        self.frame_dims = frame_dims
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.kernel_size = kernel_size

        self.convLSTM = StackedConvLSTM(
            in_channels, hidden_dim, kernel_size, num_lstm_layers
        )

        self.decoder = nn.Conv2d(hidden_dim, in_channels, kernel_size=(1, 1))
        self.activation = nn.Sigmoid()

    def forward(self, frame_sequence: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to predict the next frame.

        Parameters
        ----------
        frame_sequence: torch.Tensor
            Input sequence of frames with shape (batch_size, num_frames, height, width)

        Returns
        -------
        torch.Tensor
            Predicted next frame with shape (batch_size, height, width)
        """
        if len(frame_sequence.shape) == 4:
            frame_sequence = frame_sequence.unsqueeze(2)
        out, _ = self.convLSTM(frame_sequence)

        last = out[:, -1]
        prediction = self.decoder(last)

        return self.activation(prediction)

    def forward_multi_step(
        self,
        frame_sequence: torch.Tensor,
        num_steps: int,
        teacher_forcing_prob: float,
        target_sequence: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict multiple future frames autoregressively.

        Parameters
        ----------
        frame_sequence: torch.Tensor
            Input sequence of frames with shape (batch_size, num_frames, height, width)
        num_steps: int
            Number of future frames to predict
        teacher_forcing_prob: float
            Probability that target is used as next input
        target_sequence: torch.Tensor
            Target sequence of frames with shape (batch_size, num_frames, heigth, width)

        Returns
        -------
        torch.Tensor
            Predicted frames with shape (batch_size, num_steps, height, width)
        """
        predictions = []

        # Ensure current_sequence has a channel dimension
        if len(frame_sequence.shape) == 4:
            current_input = frame_sequence.unsqueeze(2)
        else:
            current_input = frame_sequence

        # warmup phase
        # process the entire input history once to build the hidden state
        lstm_out, hidden_states = self.convLSTM(current_input)

        # get the last hidden state from the top layer to generate the first prediction
        # hidden_states is a list of (h, c) tuples
        # the final model output is h (hidden state) of the last layer
        last_h = hidden_states[-1][0]

        # generation phase
        for i in range(num_steps):
            # get the prediction for this frame by decoding the current hidden state
            prediction = self.activation(self.decoder(last_h))
            predictions.append(prediction.unsqueeze(1))

            # determine input for the next step (auto regressive)

            if teacher_forcing_prob is not None and np.random.random() < teacher_forcing_prob:
                next_in = target_sequence[:, i:i+1]
            else:
                next_in = prediction.unsqueeze(1)

            # update the hidden state with only the new frame
            _, hidden_states = self.convLSTM(next_in, hidden_states)

            # update last_h for the decoder
            last_h = hidden_states[-1][0]

        return torch.cat(predictions, dim=1).squeeze(2)

    def get_parameter_count(self) -> int:
        """
        Return the total number of trainable parameters in the model.
        Useful for monitoring model complexity.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
