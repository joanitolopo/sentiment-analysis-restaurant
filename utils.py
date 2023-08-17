import numpy as np

def train_val_test_split(df, train_size=0.6, val_size=0.2, random_state=42, frac=1):
  row = df.sample(frac=frac, random_state=random_state)
  train, validation, test = np.split(row, [int(train_size*len(df)), int((train_size+val_size)*len(df))])

  return train, validation, test


def tokenize(text):
    return [tok for tok in word_tokenize(text)]

