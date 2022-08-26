import sys
import numpy as np
import pickle
import time
import glob
from tqdm import tqdm
from MLTSA_sklearn.models import SKL_Train
from sklearn.neural_network import MLPClassifier

def MLTSA(data, ans, model, drop_mode="Average"):
    """

    Function to apply the Machine Learning Transition State Analysis to a given training dataset/answers and trained
    model. It calculates the Global Means and re-calculates accuracy for predicting each outcome.

    :param data: Training data used for training the ML model. Must have shape (samples, features)
    :type data: list
    :param ans: Outcomes for each sample on "data". Shape must be (samples)
    :type ans: list
    :param model: The model used for training data.
    :param drop_mode: The way of calculating accuracy drop. Default: "Average"
    :return: The accuracy after deleting one data

    """
    # Calculating the global means
    means_per_sim = np.mean(data.T, axis=1)
    gmeans = np.mean(means_per_sim, axis=1)
    temp_sim_data = np.copy(data)

    # Swapping the values and predicting for the FR
    FR = []
    for y, data in tqdm(enumerate(temp_sim_data)):
        mean_sim = []
        for n, mean in enumerate(gmeans):
            tmp_dat = np.copy(data)
            tmp_dat.T[n, :] = mean
            yy = np.zeros(len(tmp_dat)).astype(str)
            yy[:] = ans[y]
            res = model.score(tmp_dat, yy)
            mean_sim.append(res)
        FR.append(mean_sim)
    return FR

def read_dataset(ns_long_sim, ns_for_train, percent_end, filename):
    """
    This function is defined to read data from the dataset and transfer the data into the one that can be used for train
    ing.

    :param ns_long_sim: the time for each simulation (ns).
    :type ns_long_sim: float
    :param ns_for_train: The time for each training (ns).
    :type ns_for_train: float
    :param percent_end: The length of the frame in each simulation
    :type percent_end: float
    :param filename: The binary file storing CVs
    :type filename: string
    :return: The data that can be trained by models.
    """
    data = []
    print("dataset", filename)
    counter = 0
    with open(filename, "rb") as fr:
        try:
            while True:
                sim = pickle.load(fr)
                counter += 1
                frames = int((len(sim[0])/ns_long_sim)*ns_for_train)
                sim = sim[0][int(frames*(1-percent_end)):int(frames)]
                data.append(sim)
        except:
            pass
    print(counter, " simulations in dataset")
    return data

# Run the code according to the input of the command line
args = sys.argv
dataset = args[1]

traj_order = ['traj_132', 'traj_125', 'traj_165', 'traj_111', 'traj_46', 'traj_129', 'traj_43', 'traj_42', 'traj_134',
              'traj_102', 'traj_163', 'traj_197', 'traj_64', 'traj_66', 'traj_24', 'traj_56', 'traj_199', 'traj_167',
              'traj_82', 'traj_130', 'traj_141', 'traj_124', 'traj_179', 'traj_104', 'traj_140', 'traj_184', 'traj_150',
              'traj_80', 'traj_143', 'traj_145', 'traj_146', 'traj_33', 'traj_190', 'traj_157', 'traj_158', 'traj_20',
              'traj_48', 'traj_97', 'traj_10', 'traj_26', 'traj_154', 'traj_159', 'traj_21', 'traj_5', 'traj_15',
              'traj_122', 'traj_11', 'traj_178', 'traj_93', 'traj_112', 'traj_138', 'traj_41', 'traj_2', 'traj_9',
              'traj_49', 'traj_14', 'traj_192', 'traj_168', 'traj_99', 'traj_94', 'traj_13', 'traj_114', 'traj_47',
              'traj_79', 'traj_6', 'traj_98', 'traj_133', 'traj_32', 'traj_177', 'traj_4', 'traj_75', 'traj_127',
              'traj_106', 'traj_65', 'traj_191', 'traj_172', 'traj_171', 'traj_108', 'traj_182', 'traj_84', 'traj_78',
              'traj_187', 'traj_113', 'traj_31', 'traj_193', 'traj_96', 'traj_101', 'traj_71', 'traj_144', 'traj_29',
              'traj_3', 'traj_67', 'traj_109', 'traj_60', 'traj_25', 'traj_45', 'traj_76', 'traj_119', 'traj_164',
              'traj_22', 'traj_7', 'traj_174', 'traj_23', 'traj_117', 'traj_175', 'traj_185', 'traj_189', 'traj_188',
              'traj_198', 'traj_54', 'traj_70', 'traj_105', 'traj_173', 'traj_68', 'traj_156', 'traj_63', 'traj_83',
              'traj_51', 'traj_50', 'traj_115', 'traj_128', 'traj_135', 'traj_149', 'traj_72', 'traj_126', 'traj_121',
              'traj_155', 'traj_169', 'traj_116', 'traj_194', 'traj_52', 'traj_69', 'traj_73', 'traj_120', 'traj_162',
              'traj_57', 'traj_40', 'traj_95', 'traj_142', 'traj_136', 'traj_161', 'traj_81', 'traj_74', 'traj_62',
              'traj_55', 'traj_180', 'traj_176', 'traj_152', 'traj_139']

# Read the original data
bin_path = []
bins = glob.glob("CV_*_d{}.bin".format(dataset))
bins.sort()
data = []
for n, file in enumerate(bins):
    if n == 0:
        data.append(read_dataset(0.5, 0.1, 0.50, file)[:-1])
    else:
        data.append(read_dataset(0.5, 0.1, 0.50, file))
data = np.concatenate(data)
print(data.shape)
frames_per_sim = len(data[0])

# Get answers
ans_list = np.genfromtxt("in_out_list.dat", dtype=str)
Y = []
AnsList = []
for x, name in enumerate(traj_order):
    idx = np.where(ans_list.T[0] == name)[0][0]
    AnsList.append(ans_list[idx][1])
    tmp = np.zeros(frames_per_sim).astype(str)
    tmp[:] = ans_list[idx][1]
    Y.append(tmp)
print(np.array(Y).shape)

# X and Y for supervised learning
X = np.concatenate(data, axis=0)
Y = np.concatenate(Y)

# MLTSA
reps = 10
for R in range(0, reps):
    results = {}
    s = time.time()

    # The model for training
    NN = MLPClassifier(verbose=True, max_iter=500, random_state=0)
    print("Setup Classifier")

    # Mix features and train the model
    idx = np.arange(0, X.shape[1])
    np.random.shuffle(idx)
    results["idx"] = idx
    print("Mixed Features")
    X_mix = X[:, idx]
    trained_NN, train_acc, test_acc = SKL_Train(NN, X_mix[:, :], Y[:])
    print("Model Trained")

    # Apply MLTSA
    results["model"] = trained_NN
    results["scores"] = [train_acc, test_acc]
    ADrop_train_avg = MLTSA(data[:, :, idx], AnsList, trained_NN, drop_mode="Average")
    print("MLTSA done")
    results["mltsa"] = ADrop_train_avg
    infile = open("results/MLP_res_mixed_d{}_{}.bin".format(dataset, R), "wb")
    pickle.dump(results, infile)
    infile.close()

    print((time.time() - s) / 60, " minutes for this training")
