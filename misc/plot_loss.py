import matplotlib.pyplot as plt
import pandas as pd
import sys

input_csv = sys.argv[1]

csv = pd.read_csv(input_csv)

epoch = csv['epoch'].values
val_loss = csv['valid_loss'].values
train_loss = csv['train_loss'].values

plt.plot(epoch, train_loss, color='blue', label='train')
plt.plot(epoch, val_loss, color='red', label='valid')
plt.legend()
plt.show()