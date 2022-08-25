#####
import sys
import numpy as np
import pickle
import time
import glob

from numpy import ndarray
from tqdm import tqdm

def MLTSA(data, ans, model, drop_mode="Average", data_mode="Normal"):
    """
    Function to apply the Machine Learning Transition State Analysis to a given training dataset/answers and trained
    model. It calculates the Gloabl Means and re-calculates accuracy for predicting each outcome.

    :param data: Training data used for training the ML model. Must have shape (samples, features)
    :type data: list
    :param ans: Outcomes for each sample on "data". Shape must be (samples)
    :type ans: list
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

args = sys.argv
dataset = args[1]

traj_order = ['traj_132', 'traj_125', 'traj_165', 'traj_111', 'traj_46', 'traj_129', 'traj_43', 'traj_42', 'traj_134', 'traj_102', 'traj_163', 'traj_197', 'traj_64', 'traj_66', 'traj_24', 'traj_56', 'traj_199', 'traj_167', 'traj_82', 'traj_130', 'traj_141', 'traj_124', 'traj_179', 'traj_104', 'traj_140', 'traj_184', 'traj_150', 'traj_80', 'traj_143', 'traj_145', 'traj_146', 'traj_33', 'traj_190', 'traj_157', 'traj_158', 'traj_20', 'traj_48', 'traj_97', 'traj_10', 'traj_26', 'traj_154', 'traj_159', 'traj_21', 'traj_5', 'traj_15', 'traj_122', 'traj_11', 'traj_178', 'traj_93', 'traj_112', 'traj_138', 'traj_41', 'traj_2', 'traj_9', 'traj_49', 'traj_14', 'traj_192', 'traj_168', 'traj_99', 'traj_94', 'traj_13', 'traj_114', 'traj_47', 'traj_79', 'traj_6', 'traj_98', 'traj_133', 'traj_32', 'traj_177', 'traj_4', 'traj_75', 'traj_127', 'traj_106', 'traj_65', 'traj_191', 'traj_172', 'traj_171', 'traj_108', 'traj_182', 'traj_84', 'traj_78', 'traj_187', 'traj_113', 'traj_31', 'traj_193', 'traj_96', 'traj_101', 'traj_71', 'traj_144', 'traj_29', 'traj_3', 'traj_67', 'traj_109', 'traj_60', 'traj_25', 'traj_45', 'traj_76', 'traj_119', 'traj_164', 'traj_22', 'traj_7', 'traj_174', 'traj_23', 'traj_117', 'traj_175', 'traj_185', 'traj_189', 'traj_188', 'traj_198', 'traj_54', 'traj_70', 'traj_105', 'traj_173', 'traj_68', 'traj_156', 'traj_63', 'traj_83', 'traj_51', 'traj_50', 'traj_115', 'traj_128', 'traj_135', 'traj_149', 'traj_72', 'traj_126', 'traj_121', 'traj_155', 'traj_169', 'traj_116', 'traj_194', 'traj_52', 'traj_69', 'traj_73', 'traj_120', 'traj_162', 'traj_57', 'traj_40', 'traj_95', 'traj_142', 'traj_136', 'traj_161', 'traj_81', 'traj_74', 'traj_62', 'traj_55', 'traj_180', 'traj_176', 'traj_152', 'traj_139']

traj_order_cut = []
for i in range(len(traj_order)):
    if (i < 15) or (30 < i < 75) or (90 < i < 105) or (120 < i):
        traj_order_cut.append(traj_order[i])
# print(len(traj_order))
# print(len(traj_order_cut))

X = read_bin("pca_components.bin")

######## GETTING ANSWERS ############
ans_list = np.genfromtxt("in_out_list.dat", dtype=str)

Y = []
AnsList = []

for x, name in enumerate(traj_order_cut):
    idx = np.where(ans_list.T[0] == name)[0][0]
    AnsList.append(ans_list[idx][1])
    tmp = np.zeros(2500).astype(str)
    tmp[:] = ans_list[idx][1]
    Y.append(tmp)

print(np.array(Y).shape)
Y = np.concatenate(Y)
print(len(Y))


from MLTSA_sklearn.models import SKL_Train
from sklearn.ensemble import GradientBoostingClassifier
from joblib import Parallel, delayed

def train_run(R):

    results = {}

    s = time.time()

    clf = GradientBoostingClassifier(random_state=0)
    print("Setup Classifier")

    idx = np.arange(0, X.shape[1])
    results["idx"] = idx
    print("Mixed Features")

    X_mix = X[:, :100]
    trained_clf, train_acc, test_acc = SKL_Train(clf, X_mix[:, :], Y[:])
    print("Model Trained")

    results["model"] = trained_clf
    results["scores"] = [train_acc, test_acc]
    # ADrop_train_avg = MLTSA(data[:, :, idx], AnsList, trained_clf, drop_mode="Average")
    # print("MLTSA done")
    # results["mltsa"] = ADrop_train_avg

    infile = open("results/GBDT_PCA_res_mixed_{}.bin".format(R), "wb")

    pickle.dump(results, infile)
    infile.close()
    print((time.time() - s)/60, " minutes for this training")

replicas = 10
parallel_jobs = 10
result = Parallel(n_jobs=parallel_jobs)(delayed(train_run)(R) for R in range(replicas))
