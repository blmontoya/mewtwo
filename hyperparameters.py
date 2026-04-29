# Height and width of desired range image
# (in RangeNet++, they have input sizes [64 × 2048], [64 × 1024], and [64 × 512])
H_RANGE = 64
W_RANGE = 1024

# CNN Training Hyperparameters
NUM_EPOCHS = 15
BATCH_SIZE = 64
GAMMA = 1 #? 
LR = 0.001
STEP_SIZE = 0.5

# KNN Hyperparameters 
CUT_OFF = 1.0 #maximum allowed distance of a point considered a near neighbor
NBRHOOD_SIZE = 5 #fix this