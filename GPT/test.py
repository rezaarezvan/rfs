with open('./data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


print("Length of dataset in characters: ", len(text))

# All uniqu characters in the file
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[ch] for ch in s]
def decode(x): return ''.join([itos[i] for i in x])


print("Encoded: ", encode("Hello World"))
print("Decoded: ", decode(encode("Hello World")))
