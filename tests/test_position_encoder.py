import unittest
import torch
from src.position_encoder import PositionEncoding


class TestPositionEncoding(unittest.TestCase):

    def setUp(self):
        self.d_model = 2
        self.max_len = 6
        self.position_encoding = PositionEncoding(d_model=self.d_model, max_len=self.max_len)

    def test_pe_shape(self):
        # Check the shape of the position encoding matrix
        self.assertEqual(self.position_encoding.pe.shape, (self.max_len, self.d_model))

    def test_forward_shape(self):
        # Create a dummy word token embedding of shape (max_len, d_model)
        wte = torch.randn(self.max_len, self.d_model)
        output = self.position_encoding(wte)
        self.assertEqual(output.shape, (self.max_len, self.d_model))

    def test_forward_values(self):
        # Create a dummy word token embedding of shape (max_len, d_model)
        wte = torch.zeros(self.max_len, self.d_model)
        output = self.position_encoding(wte)
        
        # The output should match the position encoding values since wte is zero
        self.assertTrue(torch.allclose(output, self.position_encoding.pe[:self.max_len, :]))

    def test_register_buffer(self):
        # Check if 'pe' is registered as a buffer and not a parameter
        self.assertFalse('pe' in dict(self.position_encoding.named_parameters()))
        self.assertTrue('pe' in dict(self.position_encoding.named_buffers()))

if __name__ == '__main__':
    unittest.main()