import numpy as np
from tqdm import tqdm


class HopfieldNetwork(object):
    def fit(self, train_data):
        print("Start to train weights...")
        num_data = len(train_data)
        self.num_neuron = train_data[0].shape[0]

        W = np.zeros((self.num_neuron, self.num_neuron))
        rho = np.sum([np.sum(t) for t in train_data]) / (num_data * self.num_neuron)

        for i in tqdm(range(num_data)):
            t = train_data[i] - rho
            W += np.outer(t, t)

        diagW = np.diag(np.diag(W))
        W = W - diagW
        W /= num_data

        self.W = W

    def predict(self, data, num_iter=20, threshold=0):
        print("Start to predict...")
        self.num_iter = num_iter
        self.threshold = threshold

        copied_data = np.copy(data)

        predicted = []
        for i in tqdm(range(len(data))):
            predicted.append(self._run(copied_data[i]))
        return predicted

    def _run(self, init_s):
        """
        Synchronous update
        """

        s = init_s
        e = self.energy(s)

        for _ in range(self.num_iter):
            s = np.sign(self.W @ s - self.threshold)
            e_new = self.energy(s)

            if e == e_new:
                return s

            e = e_new
        return s

    def energy(self, s):
        return -0.5 * s @ self.W @ s + np.sum(s * self.threshold)
