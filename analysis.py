import numpy as np
import matplotlib.pyplot as plt
import pickle

UniRes = []
for i in range(10):
    reps = 10
    adrops = []
    #####Fetching the data from the binary files #####
    for R in range(0, reps):
        # results = pickle.load(open("results/MLP_res_mixed_d{}_{}.bin".format(i,R), "rb"))
        results = pickle.load(open("results/GBDT_res_mixed_d{}_{}.bin".format(i, R), "rb"))
        adrop = results["mltsa"]
        idx = results["idx"]
        model = results["model"]
        sort = np.argsort(idx)
        adrops.append(np.mean(np.array(adrop).T[sort], axis=1))

    #### Analyzing the data to get the top features of each replica #####
    tops = []
    top_size = 10
    for rep in adrops:
        idx = (rep).argsort()[:top_size]
        tops.append(list(idx))
    mx = len(tops)

    from collections import Counter
    tops = np.concatenate(tops)
    top_features = Counter(list(tops))

    ### Intepreting which features are top ####
    # print("Top features are", top_features.most_common(5))
    selection_strings = np.load("string_selections_GPCR_{}.npy".format(i))
    with open("adrop_top_GBDT_{}.txt".format(i), "w") as f:
        for feature, t in top_features.most_common():
            UniRes.append(selection_strings[feature][0][7:11] + selection_strings[feature][0][23:27])
            UniRes.append(selection_strings[feature][1][7:11] + selection_strings[feature][1][23:27])
            f.write("{}\t{}\t{}\t{}\n".format(feature, np.around((t/mx)*100), selection_strings[feature][0],selection_strings[feature][1]))

    ######## Plot the MLTSA results (Accuracy Drop) #######
    plt.figure(figsize=(6,3))
    plt.title("Mean Accuracy Drop")
    plt.plot(np.mean(adrops, axis=0),"-s", markersize=3, color="black", label="Mean Adrop")
    for feature, t in top_features.most_common(5):
        plt.scatter(feature, np.mean(adrops, axis=0)[feature],
                    label="Top feature {}, {}%".format(feature, np.around((t/mx)*100, decimals=2)), color="r", alpha=(t/mx))
    plt.ylabel("Mean Accuracy")
    plt.xlabel("Feature")
    plt.legend()
    plt.savefig("string_selections_GPCR__{}.png".format(i))
    plt.show()
    print("Feature Accuracy Drop Top:", np.argmin(np.mean(adrops, axis=0)))

UniResCount = Counter(UniRes)
UniResType = [x[0:4] for x in UniRes]
UniResType = Counter(UniResType)
print(UniResType)
print(UniResCount)

ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.pie([float(v) for v in UniResType.values()], labels=[l for l in UniResType], autopct='%1.1f%%')
ax2.pie([float(v) for v in UniResCount.values()], labels=[l for l in UniResCount], autopct='%1.1f%%')
ax1.set_title("Type of residue involved")
ax2.set_title("Residue involved at top")
plt.tight_layout()
# plt.savefig("pie1_10top.svg")
# plt.savefig("MLP_pie1_10top.jpg")
plt.savefig("GBDT_pie1_10top.jpg")
plt.show()



ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.pie([float(v) for v in UniResType.values()][1:], labels=[l for l in UniResType][1:], autopct='%1.1f%%')
ax2.pie([float(v) for v in UniResCount.values()][1:], labels=[l for l in UniResCount][1:], autopct='%1.1f%%')
ax1.set_title("Type of residue involved")
ax2.set_title("Residue involved at top")
plt.tight_layout()
# plt.savefig("pie2_10top.svg")
# plt.savefig("MLP_pie2_10top.jpg")
plt.savefig("GBDT_pie2_10top.jpg")
plt.show()


print()