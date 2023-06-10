src_vocab_size = 5000  # Source vocabulary size
tgt_vocab_size = 5000  # Target vocabulary size
d_model = 512          # Embedding size
num_heads = 8          # Number of heads in multi-head attention
num_layers = 6         # Number of layers in encoder and decoder
d_ff = 2048            # Hidden layer size in feed forward network inside transformer
max_seq_length = 100   # Maximum sequence length
dropout = 0.1          # Dropout rate
epochs = 100           # Number of epochs

lr = 1e-4              # Learning rate
betas = (0.9, 0.98)    # Betas for Adam optimizer
eps = 1e-9             # Epsilon for Adam optimizer
