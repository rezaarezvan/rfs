# parameters.py

# Dataset parameters
IMG_SIZE = 32  # size of the images in the CIFAR-10 dataset
PATCH_SIZE = 4  # size of the patches to break the images into
NUM_CLASSES = 10  # number of classes in the CIFAR-10 dataset

# Model parameters
EMBED_DIM = 64  # dimension of the patch embeddings
NUM_HEADS = 4  # number of attention heads
NUM_LAYERS = 4  # number of transformer layers

# Training parameters
BATCH_SIZE = 64  # number of images per batch
LEARNING_RATE = 1e-3  # learning rate for the optimizer
NUM_EPOCHS = 10  # number of times to iterate over the entire dataset
