import torch
from torch.utils.data import TensorDataset, DataLoader

from src.transformer import DecoderOnlyTransformer
import pytorch_lightning as L


# Define a very small vocab
token_to_id = {'what' : 0,
               'is' : 1,
               'your' : 2,
               'name': 3,
               'Tony': 4,
               '<EOS>' : 5,
              }
id_to_token = dict(map(reversed, token_to_id.items()))

# Prepare training dataset
inputs = torch.tensor([[token_to_id["what"],
                        token_to_id["is"], 
                        token_to_id["your"],
                        token_to_id["name"],
                        token_to_id["<EOS>"],
                        token_to_id["Tony"]], 

                       [token_to_id["your"],
                        token_to_id["name"],
                        token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["Tony"]]])

labels = torch.tensor([[token_to_id["is"],
                        token_to_id["your"],
                        token_to_id["name"],
                        token_to_id["<EOS>"],
                        token_to_id["Tony"],
                        token_to_id["<EOS>"]], 

                        [token_to_id["name"],
                        token_to_id["is"], 
                        token_to_id["what"], 
                        token_to_id["<EOS>"], 
                        token_to_id["Tony"],
                        token_to_id["<EOS>"]]])

train_dataset = TensorDataset(inputs, labels)
dataloader = DataLoader(train_dataset)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Training scripts
model = DecoderOnlyTransformer(num_tokens=len(token_to_id), d_model=2, max_len=8)

# By setting accelerator to gpu, Lightning will automatically load model and tranining inputs on GPU
# however make sure the intermediate tensors created on-the-fly in forward() function still 
# needs to be moved to device.
trainer = L.Trainer(max_epochs=30, accelerator="gpu", devices=1)
trainer.fit(model, train_dataloaders=dataloader)

# Validate
print("Validation:\n")
model_input = torch.tensor([token_to_id["what"], 
                            token_to_id["is"], 
                            token_to_id["your"],
                            token_to_id["name"], 
                            token_to_id["<EOS>"]])
input_length = model_input.size(dim=0)
predictions = model(model_input)
print(torch.argmax(predictions[-1,:]))
