import math
import unittest
from typing import cast

import torch

from eurocropsssl.models.positional_encoding import PositionalEncoding


class TestPositionalEncoding(unittest.TestCase):
    def setUp(self) -> None:
        # parameters for testing
        self.d_hid = 128
        self.pos_enc_len = 366
        self.batch_size = 16
        self.seq_len = 105
        self.padding_value = -1

        # Create default PositionalEncoding instance
        self.pos_enc = PositionalEncoding(
            d_hid=self.d_hid,
            pos_enc_len=self.pos_enc_len,
            padding_value_dates=self.padding_value,
        )
        # create dates
        rand = torch.rand(self.batch_size, self.pos_enc_len)
        _, indices = torch.sort(rand, dim=1)
        self.dates, _ = indices[:, : self.seq_len].sort()

    def test_initialization(self) -> None:
        """Test if the PositionalEncoding initializes correctly."""
        # check if p has the correct shape
        self.assertEqual(self.pos_enc.p.shape, (self.pos_enc_len, self.d_hid))

        # check if padding value is set correctly
        self.assertEqual(self.pos_enc.padding_value_dates, self.padding_value)

        # check if sine and cosine values are properly initialized
        # first column (index 0) should have sine values
        # second column (index 1) should have cosine values
        timesteps = torch.arange(self.pos_enc_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_hid, 2).float() * (-math.log(1000.0) / self.d_hid)
        )

        # check a few values to ensure correct initialization
        expected_sine = torch.sin(timesteps * div_term)[0, 0]
        expected_cosine = torch.cos(timesteps * div_term)[0, 0]

        self.assertAlmostEqual(
            float(cast(torch.Tensor, self.pos_enc.p)[0, 0]),
            float(expected_sine),
            places=6,
        )
        self.assertAlmostEqual(
            float(cast(torch.Tensor, self.pos_enc.p)[0, 1]),
            float(expected_cosine),
            places=6,
        )

    def test_forward_add_to_x(self) -> None:
        """Test forward pass with add_to_x=True (default)."""
        # create input tensors
        x = torch.rand(self.batch_size, self.seq_len, self.d_hid)

        # store original x for comparison
        x_original = x.clone()

        # forward pass
        output = self.pos_enc(x, self.dates)

        # check if output shape matches input shape
        self.assertEqual(output.shape, x.shape)

        # check if values differ from input (indicating that positional encoding was added)
        self.assertFalse(torch.allclose(output, x_original))

    def test_forward_no_add_to_x(self) -> None:
        """Test forward pass with add_to_x=False."""
        # create input tensors
        x = torch.rand(self.batch_size, self.seq_len, self.d_hid)

        # forward pass without adding to x
        output = self.pos_enc(x, self.dates, add_to_x=False)

        # check if output shape matches input shape
        self.assertEqual(output.shape, x.shape)

        # the output should be the positional encoding only, not added to x
        self.assertFalse(torch.allclose(output, x))

    def test_with_batch_padding(self) -> None:
        """Test forward pass with padded date values."""
        # create input tensors with some padding
        x = torch.rand(self.batch_size, self.seq_len, self.d_hid)

        dates = self.dates.clone()
        # set some values to padding value
        dates[1:2, self.seq_len - 4] = self.padding_value
        dates[15:, self.seq_len - 10] = self.padding_value

        # forward pass
        output = self.pos_enc(x, dates)

        # check that padded positions have same values as original x
        # (no positional encoding added to padded positions)
        padded_mask = dates.eq(self.padding_value)
        padded_positions = padded_mask.unsqueeze(-1).expand_as(x)

        # extract values at padded positions from output and original x
        output_padded = output[padded_positions].view(-1)
        x_padded = x[padded_positions].view(-1)

        # check if they're the same (no positional encoding added)
        self.assertTrue(torch.allclose(output_padded, x_padded))

    def test_padded_sequence_to_366(self) -> None:
        """Test the case where dates.size(-1) < x.size(-2)."""
        x = torch.rand(self.batch_size, self.pos_enc_len + 5, self.d_hid)

        # forward pass
        output = self.pos_enc(x, self.dates)

        # check output shape
        self.assertEqual(output.shape, x.shape)

    def test_custom_period(self) -> None:
        """Test with a custom period parameter."""
        # create a position encoding with a different period
        custom_t = 500
        pos_enc_custom = PositionalEncoding(
            d_hid=self.d_hid,
            pos_enc_len=self.pos_enc_len,
            t=custom_t,
            padding_value_dates=self.padding_value,
        )

        # create input tensors
        x = torch.rand(self.batch_size, self.seq_len, self.d_hid)

        # forward pass
        output = pos_enc_custom(x, self.dates)

        # should differ from default period output
        output_default = self.pos_enc(x, self.dates)
        self.assertFalse(torch.allclose(output, output_default))

    def test_different_add_to_x_parameter(self) -> None:
        """Test both values of the add_to_x parameter."""
        # create input tensors
        x = torch.rand(self.batch_size, self.seq_len, self.d_hid)

        # get positional encodings without adding to x
        p_only = self.pos_enc(x, self.dates, add_to_x=False)

        # get result with adding to x
        result_with_add = self.pos_enc(x, self.dates, add_to_x=True)

        # valculate manually adding p to x
        manual_add = x + p_only

        # both should be the same
        self.assertTrue(torch.allclose(result_with_add, manual_add))
