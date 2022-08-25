import numpy as np
import matplotlib.pyplot as plt
import pickle

def read_bin(filename):
    data = []
    with open(filename,"rb") as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data

def create_bin(filename, data):
    file = open(filename, "wb+")
    pickle.dump(data, file)
    return file

def append_bin(file, data):
    pickle.dump(data, file)
    return

def read_dataset(ns_long_sim, ns_for_train, percent_end, filename):
    data = []
    print("dataset", filename)
    counter = 0
    with open(filename, "rb") as fr:
        try:
            while True:
                sim = pickle.load(fr)
                counter += 1
                frames = int((len(sim)/ns_long_sim)*ns_for_train)
                length0 = int(frames*(1-percent_end))
                length1 = int(frames)
                sim = sim[length0:length1]
                data.append(sim)
        except EOFError:
            pass
    print(counter, " simulations in dataset")
    print(np.array(data).shape)
    return data

bin_path = []
bins = [
"CV_0_50.bin",
"CV_50_100.bin",
"CV_100_150.bin"
]

data = []
for n, file in enumerate(bins):
    data.append(read_dataset(0.5, 0.1, 0.5, file))
data = np.concatenate(data)
print(data.shape)
frames_per_sim = len(data[0])

X = np.concatenate(data, axis=0)
print("X shape is ", X.shape)

from sklearn.decomposition import PCA

pca = PCA()
pca.fit(X)
print("PCA model fitted")

plt.figure()
plt.plot((pca.explained_variance_ratio_)) 
plt.savefig("pca_explained_variance.svg")

filesave = open("pca_fitted.pca", "wb")
pickle.dump([pca], filesave)
print("PCA model saved")

X_pca = pca.transform(X)
filesave = open("pca_components.bin", "wb")
pickle.dump(X_pca, filesave)
print("New features saved in bin")