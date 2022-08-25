
import numpy as np
import CV_from_MD as cvs
import pickle
import mdtraj as md
import time
import glob
from tqdm import tqdm

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

# Testing the labelling system
dcd_path = []
top_path = []
folders = glob.glob("../traj_*/")
print(folders)
print(len(folders))

for f in folders:
    dcd_path.append(str(f)+"string_cropped.dcd")
    top_path.append(str(f)+"string_cropped.pdb")

freq = 50
st = time.time()
all_CV = []
for i in tqdm(range(len(dcd_path))):
    r = time.time()
    traj = md.iterload(dcd_path[i], top=top_path[i], chunk=25000)
    traj = next(traj)
    idx = traj.top.select("not element H and protein")
    print(len(idx))
    xyz = traj.xyz[:, idx]
    xyz = np.concatenate(xyz.T, axis=0)

    sel_strings = []
    for id in idx:
        a = traj.top.atom(id)
        label = [str(a) + "-X", str(a) + "-Y", str(a) + "-Z"]
        sel_strings.append(label)
    sel_strings = np.array(sel_strings)
    print(sel_strings.shape)
    sel_strings = np.concatenate(sel_strings, axis=0)
    np.save("selection_strings_XYZ_nolig.npy", sel_strings)

    print(xyz.shape)
    distances = xyz.T
    print("CVs fetched for", dcd_path[i], "in ", (time.time() - r)/60, "minutes")
    # Create bin files for CVs
    if i == 0:
        filename = "CV_" + str(0) + "_" + str(freq) + ".bin"
        obj = create_bin(filename, distances)
    elif i % freq == 0 and i != 0:
        filename = "CV_" + str(i) + "_" + str(i+freq) + ".bin"
        obj = create_bin(filename, distances)
    else:
        append_bin(obj, distances)

print("Dataset created in ", (time.time() - st)/60, " minutes")


