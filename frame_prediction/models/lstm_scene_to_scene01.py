"""
LSTM Scene-to-Scene Decoder
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


class LSTMSceneToScene01(nn.Module):
    """
    LSTM-based model for predicting the next frame in a 2D dynamic scene consisting of a sequence of frames.

    Architecture:
        - Encoder: flatten each frame
        - LSTM to model temporal dependencies
        - Decoder: linear layer to predict next frame
    """

    frame_dims: Tuple[int, int]
    hidden_dim: int
    num_lstm_layers: int

    def __init__(
        self,
        frame_dims: Tuple[int, int] = (64, 64),
        hidden_dim: int = 512,
        num_lstm_layers: int = 2,
        dropout_rate: float = 0.2,
    ):
        """
        Initialize the LSTM scene-to-scene prediction model.

        Parameters
        ----------
        frame_dims: Tuple[int, int]
            Dimensions of each frame (height, width)
        hidden_dim: int
            Hidden dimension of the LSTM layers
        num_lstm_layers: int
            Number of stacked LSTM layers
        dropout_rate: float
            Dropout probability for regularization
        """
        super(LSTMSceneToScene01, self).__init__()

        # set attributes
        self.frame_dims = frame_dims
        self.hidden_dim = hidden_dim
        self.num_lstm_layers = num_lstm_layers
        self.dropout_rate = dropout_rate

        # calculate flattened frame size
        self.frame_size = frame_dims[0] * frame_dims[1]

        # encoder: flatten frame (identity operation, kept for interface consistency)
        self.encoder = nn.Flatten(start_dim=-2, end_dim=-1)

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.frame_size,
            hidden_size=hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout_rate if num_lstm_layers > 1 else 0.0,
        )

        # decoder: LSTM hidden state to predicted frame
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, self.frame_size),
            nn.Sigmoid(),
        )

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
        batch_size, num_frames, height, width = frame_sequence.shape

        # encode: flatten each frame (batch, num_frames, H, W) -> (batch, num_frames, H * W)
        encoded = self.encoder(frame_sequence)

        # pass through LSTM
        lstm_out, _ = self.lstm(encoded)

        # take the last LSTM output for prediction
        lstm_last = lstm_out[:, -1, :]

        # decode: predict next frame
        predicted_flat = self.decoder(lstm_last)

        # reshape to frame dimensions
        predicted_frame = predicted_flat.view(batch_size, height, width)

        return predicted_frame

    def forward_multi_step(
        self,
        frame_sequence: torch.Tensor,
        num_steps: int,
        teacher_forcing_prob: float = 0.0,
        target_sequence: torch.Tensor | None = None,
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
            Probability that target is used as next input (default: 0.0, pure autoregressive)
        target_sequence: torch.Tensor | None
            Target sequence of frames with shape (batch_size, num_steps, height, width).
            Required when teacher_forcing_prob > 0.

        Returns
        -------
        torch.Tensor
            Predicted frames with shape (batch_size, num_steps, height, width)
        """
        if teacher_forcing_prob > 0.0 and target_sequence is None:
            raise ValueError("target_sequence is required when teacher_forcing_prob > 0")

        predictions = []
        current_sequence = frame_sequence.clone()

        for i in range(num_steps):
            # predict next frame
            next_frame = self.forward(current_sequence)
            predictions.append(next_frame.unsqueeze(1))

            # determine input for the next step
            if teacher_forcing_prob > 0.0 and np.random.random() < teacher_forcing_prob:
                next_input = target_sequence[:, i]
            else:
                next_input = next_frame

            # update sequence: drop oldest frame, append next input
            current_sequence = torch.cat(
                [
                    current_sequence[:, 1:, :, :],
                    next_input.unsqueeze(1),
                ],
                dim=1,
            )

        return torch.cat(predictions, dim=1)

    def get_parameter_count(self) -> int:
        """
        Return the total number of trainable parameters in the model.
        Useful for monitoring model complexity.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
