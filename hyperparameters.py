# Height and width of desired range image
# (in RangeNet++, they have input sizes [64 × 2048], [64 × 1024], and [64 × 512])
H_RANGE = 64
W_RANGE = 512

# CNN Training Hyperparameters
NUM_EPOCHS = 2 #they ran 150 epochs in the paper
BATCH_SIZE = 1
GAMMA = 0.99 #this was the value used in the paper
LR = 0.001 #LR of 0.001 in the paper
STEP_SIZE = 1 #LR decays every epoch, as per paper

# KNN Hyperparameters 
CUT_OFF = 1.0 #maximum allowed distance of a point considered a near neighbor
NBRHOOD_SIZE = 5 #fix this