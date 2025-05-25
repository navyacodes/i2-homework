import numpy as np
import matplotlib.pyplot as plt
import sklearn
import torchvision as tv

all_data = tv.datasets.MNIST('./data', download=True)

data = all_data.data.numpy()
labels = all_data.targets.numpy()

print(f"Data Shape:{np.shape(data)}") # = (60000, 28, 28)
print(f"Label Shape: {np.shape(labels)}") # = (60000,)

n_train = 10000
n_test = 500

train_data = data[:n_train]
train_labels = labels[:n_train]

test_data = data[-n_test: ]
test_labels = labels[-n_test: ]

print(f"Train Data Shape:{train_data.shape}") # = (n_train, 28, 28)
print(f"Train Label Shape: {train_labels.shape}") # = (n_train,)

print(f"Test Data Shape:{test_data.shape}") # = (n_test, 28, 28)
print(f"Test Label Shape: {test_labels.shape}") # = (n_test,)

num_of_digits_to_viz = 3
for i in range(num_of_digits_to_viz):
    to_reshape = train_data[i]
    plt.matshow(to_reshape.reshape(28, 28), cmap='Greys')
    plt.show()
    print(f"Associated Label: {train_labels[i]}")
    print("---")


y = np.zeros((train_labels.shape[0], 10))
y[np.arange(train_labels.shape[0]), train_labels] = 1

y_test = np.zeros((test_labels.shape[0], 10))
y_test[np.arange(test_labels.shape[0]), test_labels] = 1

X = np.reshape(train_data, [n_train, 784])
X_test = np.reshape(test_data, [n_test, 784])

X = np.hstack((X, np.ones((X.shape[0], 1))))
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

lambda_reg = 0.001

def solve_rr(X, y):
  pass

  w = np.linalg.inv(np.matmul(X.T, X) + (lambda_reg * np.eye(X.shape[1]))) @ X.T @ y
  return w

w = solve_rr(X, y)

total = 0
correct = 0
for x, label in zip(X_test, y_test):
  pred = np.argmax(w.T @ x)
  gt = np.argmax(label)
  if pred == gt:
    correct += 1
  total += 1

import time
t1 = time.time()
print(f"Accuracy on {len(X_test)} test examples: {correct/total}")
print(f"Time taken: {time.time()-t1}")
#----------------------------------------------------------------#
