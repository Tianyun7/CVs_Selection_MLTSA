import numpy as np
import CV_from_MD as cvs
import MLTSA.MLTSA_datasets.MD_DATA.CV_from_MD as cvs
import pickle
import mdtraj as md
import time
import glob
from tqdm import tqdm

# Binary files creating function
def create_bin(filename, data):
    file = open(filename, "wb+")
    pickle.dump(data, file)
    return file

# Binary files writing function
def append_bin(file, data):
    pickle.dump(data, file)
    return

# Testing the labelling system
dcd_path = []
top_path = []
folders = glob.glob("../traj_*/")

for f in folders:
    dcd_path.append(str(f)+"string_cropped.dcd")
    top_path.append(str(f)+"string_cropped.pdb")

datasets = 10
for s in range(datasets):
    selection_strings = np.load("string_selections_GPCR_{}.npy".format(s))
    analyzer = cvs.MDs()
    freq = 50
    st = time.time()
    all_CV = []
    for i in tqdm(range(len(dcd_path))):
        try:
            r = time.time()
            # Define CVs
            CV_system = cvs.CVs(top_path[i])
            CV_system.define_variables("custom_selection", custom_selection_string=list(selection_strings))
            print("CVs defined")
            distances = analyzer.calculate_CVs(CV_system, [dcd_path[i]], loading="iterload")
            print("CVs calculated for", dcd_path[i], "in ", (time.time() - r)/60, "minutes")
            # Create binary files to store CVs
            if i == 0:
                filename = "CV_" + str(0) + "_" + str(freq) + "_d" + str(s) + ".bin"
                obj = create_bin(filename, distances)
            if i % freq == 0 and i != 0:
                filename = "CV_" + str(i) + "_" + str(i+freq) + "_d" + str(s) + ".bin"
                obj = create_bin(filename, distances)
            else:
                append_bin(obj, distances)
        except:
            print("Failed on the ", dcd_path[i], "something went wrong")
    print("Dataset created in ", (time.time() - st)/60, " minutes")
