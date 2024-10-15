### Transformers

### Roadplan

- [ ] Use Wikipedia dataset
- [ ] Refactor

### Background

Transformers use **attention**. Attention in AI,
specifically neural networks, is a technique to replicate cognitive attention.

Since, it's quite natural to focus on the "important" parts and filter out the unnecessary information.

But how do we decide which information is "important" for a neural network?

Of course this depends on the context - but given a context this is trained with **gradient descent**.


#### Transformers background

Let's firstly think of our transformer model as a black box.

We can view this black box as a box with two components - an encoder and decoder.

For example in NLP applications, transformers can be used to translate between different languages.

The encoder takes the input languages, captures it into a vector (called a *context* vector)
- the decoder, decodes the context vector item for item.

Both the encoder and decoder are RNNs. (Recurrent Neural Networks).


### Acknowledgements
The code is based on the paper "Attention Is All You Need" Ashish Vaswani et al. (2017).
