DEVICE = torch.device("cuda")
EPOCHS = 25
BATCH_SIZE = 128
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5
EARLY_STOPPING_STEPS = 10
EARLY_STOP = False

len_features=len(num_features)
len_targets=len(target_cols)
hidden_size=1024