from tqdm import tqdm
from torch import nn, optim
from data.preprocess import get_dataset
from models.transformer import Transformer
from config import *


transformer = Transformer(src_vocab_size, tgt_vocab_size, d_model,
                          num_heads, num_layers, d_ff, max_seq_length, dropout)

src_data, tgt_data = get_dataset()
criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(transformer.parameters(), lr=lr, betas=betas, eps=eps)
transformer.train()

pbar = tqdm(total=epochs, desc="Training", position=0,
            leave=True, ncols=100, unit=" epoch(s)")

for epoch in range(epochs):
    optimizer.zero_grad()
    output = transformer(src_data, tgt_data[:, :-1])
    loss = criterion(output.contiguous().view(-1, tgt_vocab_size),
                     tgt_data[:, 1:].contiguous().view(-1))
    loss.backward()
    optimizer.step()

    pbar.set_description(f"Epoch: {epoch+1}, Loss: {loss.item()}")
    pbar.update()

pbar.close()
