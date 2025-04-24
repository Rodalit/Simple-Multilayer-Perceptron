import numpy as np

class SimpleMLP:
    def __init__(self, i_dim: int = 1, h_dim: int = 16, o_dim: int = 3, n_iters: int = 500, lr: float = 2e-4, loss: str = "binary"):

        self.i_dim = i_dim
        self.h_dim = h_dim
        self.o_dim = o_dim

        self.n_iters = n_iters
        self.lr = lr

        self.loss = loss

        self.w1 = np.random.rand(i_dim, h_dim) * np.sqrt(2.0 / i_dim)
        self.b1 = np.zeros((1, h_dim))

        self.w2 = np.random.rand(h_dim, o_dim) * np.sqrt(2.0 / h_dim)
        self.b2 = np.zeros((1, o_dim))

    def relu(self, x):
        return np.maximum(0, x)

    def drelu(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        eps = np.exp(x - np.max(x))
        return eps / np.sum(eps)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def one_hot(self, y, num_classes):
        y_full = np.zeros((1, num_classes))
        y_full[0, y] = 1
        return y_full

    def categorical_cross_entropy(self, y_hat, y):
        y_hat = y_hat.reshape((1, -1))
        return -np.log(y_hat[0, y])

    def binary_crossentropy(self, y_hat, y, eps: float = 1e-15):
        y_hat = np.clip(y_hat, eps, 1 - eps)
        return - np.mean(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    def forward(self, x):
        predictions = []
        for x_i in x:
            t1 = np.dot(x_i, self.w1) + self.b1
            act = self.relu(t1)
            t2 = np.dot(act, self.w2) + self.b2
            if self.loss == "categorical":
                out = self.softmax(t2)
                predictions.append(np.argmax(out))
            elif self.loss == "binary":
                out = self.sigmoid(t2)
                predictions.append(np.where(out >= 0.5, 1, 0)[0])
        return predictions

    def train(self, X, y):
        for epoch in range(1, self.n_iters + 1):
            epoch_loss = []
            for x_i, y_i in zip(X, y):

                # forward pass
                t1 = np.dot(x_i, self.w1) + self.b1
                act = self.relu(t1)
                t2 = np.dot(act, self.w2) + self.b2
                if self.loss == "categorical":
                    out = self.softmax(t2)
                    y_hat = self.one_hot(y_i, self.o_dim)
                    error = self.categorical_cross_entropy(out, y_i)
                elif self.loss == "binary":
                    z = self.sigmoid(t2)
                    # out = np.where(z >= 0.5, 1, 0)
                    error = self.binary_crossentropy(z, y_i)

                epoch_loss.append(error)

                # backpropagation
                if self.loss == "categorical":
                    dE_dt2 = out - y_hat
                else:
                    dE_dt2 = z - y_i

                dE_dW2 = act.T @ dE_dt2
                dE_db2 = np.mean(dE_dt2, axis=0)

                dE_dh1 = dE_dt2 @ self.w2.T
                dE_dt1 = dE_dh1 * self.drelu(t1)
                dE_dW1 = np.expand_dims(x_i, axis=0).T @ dE_dt1
                dE_db1 = np.mean(dE_dt1, axis=0)

                # update weights
                self.w1 -= self.lr * dE_dW1
                self.b1 -= self.lr * dE_db1
                self.w2 -= self.lr * dE_dW2
                self.b2 -= self.lr * dE_db2

            avg_loss = np.mean(epoch_loss)
            print(f"Epoch: [{epoch}/{self.n_iters}], Loss: {avg_loss:.4f}")