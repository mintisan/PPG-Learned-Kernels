
import numpy as np
import os #path/directory stuff

base = "PPG Data" #Base directory for the PPG data
subdirs = [ #sub dirs that contain each PPG dataset
"new_PPG_DaLiA_test/processed_dataset",
"new_PPG_DaLiA_train/processed_dataset",
"TROIKA_channel_1/processed_dataset",
"WESAD_all/processed_dataset"]


X_train = np.load(os.path.join(base, subdirs[1], "scaled_ppgs.npy"))
Y_train = np.load(os.path.join(base, subdirs[1], "seg_labels.npy"))

#The rest of these datasets are test
DaLiA_X = np.load(os.path.join(base, subdirs[0], "scaled_ppgs.npy"))
DaLiA_Y = np.load(os.path.join(base, subdirs[0], "seg_labels.npy"))

TROIKA_X = np.load(os.path.join(base, subdirs[2], "scaled_ppgs.npy"))
TROIKA_Y = np.load(os.path.join(base, subdirs[2], "seg_labels.npy"))

WESAD_X = np.load(os.path.join(base, subdirs[3], "scaled_ppgs.npy"))
WESAD_Y = np.load(os.path.join(base, subdirs[3], "seg_labels.npy"))
