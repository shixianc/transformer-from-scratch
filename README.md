# Decoder-Only Transformer with PyTorch and PyTorch Lightning

This project implements a decoder-only transformer using pure PyTorch and PyTorch Lightning. The implementation includes all necessary components such as word token embeddings (WTE), word position embeddings (WPE), and attention classes.

## Project Structure

- **word_token_embedding.py**: Contains the implementation of the word token embedding (WTE) class.
- **word_position_embedding.py**: Contains the implementation of the word position embedding (WPE) class.
- **attention.py**: Contains the implementation of the attention mechanism.
- **decoder.py**: Contains the implementation of the decoder-only transformer.
- **train.py**: Script for training the transformer using PyTorch Lightning.
- **utils.py**: Utility functions used across the project.

## Dependencies

- Python 3.6 or higher
- PyTorch
- PyTorch Lightning

You can install the required dependencies using `pip`:

```sh
pip install torch pytorch-lightning
