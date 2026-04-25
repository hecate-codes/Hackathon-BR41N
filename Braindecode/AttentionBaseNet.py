from sklearn.utils import shuffle
import numpy as np
from mne.filter import filter_data

MODEL_NAME = "AttentionBaseNet"
PATH_CM = "Braindecode/CM/"

# -------- DATOS
print("Loading data...")

PATH_TRAIN = "./datos_procesados/P2_post_training.mat.npz"
PATH_TEST = "./datos_procesados/P2_post_test.mat.npz"

data_train = np.load(PATH_TRAIN)
data_test = np.load(PATH_TEST)

X_train, y_train = data_train["X"], data_train["y"]
X_test, y_test = data_test["X"], data_test["y"]

X_train = X_train.transpose(0, 2, 1)
X_test = X_test.transpose(0, 2, 1)

X_train = filter_data(X_train, sfreq=256, l_freq=8, h_freq=30)
X_test = filter_data(X_test, sfreq=256, l_freq=8, h_freq=30)


print(f"Train size: X{X_train.shape}, y{y_train.shape}")
print(f"Test size : X{X_test.shape}, y{y_test.shape}")

print("Data loaded.")
print("-"*30)

# -------- MODELO --
print("Creating model...")

import torch

from braindecode.models import AttentionBaseNet
from braindecode.util import set_random_seeds

cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
device = "cuda" if cuda else "cpu"
if cuda:
    torch.backends.cudnn.benchmark = True
seed = 20200220
set_random_seeds(seed=seed, cuda=cuda)

n_classes = 2
classes = list(range(n_classes))
# Extract number of chans and time steps from dataset
n_channels = 16
n_times = 384

model = AttentionBaseNet(
    n_chans=n_channels,
    n_outputs=n_classes,
    n_times=n_times,
)

# Display torchinfo table describing the model
print(model)

# Send model to GPU
if cuda:
    model.cuda()

print("Model created.")
print("-"*30)

# -------- ENTRENAMIENTO --

print("Training model...")
from skorch.callbacks import LRScheduler

from braindecode import EEGClassifier
from skorch.callbacks import EarlyStopping

lr = 0.001
weight_decay = 0.0001
batch_size = 64
n_epochs = 500

clf = EEGClassifier(
    model,
    criterion=torch.nn.CrossEntropyLoss,
    optimizer=torch.optim.AdamW,
    train_split=None,
    optimizer__lr=lr,
    optimizer__weight_decay=weight_decay,
    batch_size=batch_size,
    callbacks=[
        "accuracy",
        ("lr_scheduler", LRScheduler("CosineAnnealingLR", T_max=n_epochs - 1)),
    ],
    device=device,
    classes=classes,
    max_epochs=n_epochs,
)
# Model training for a specified number of epochs. `y` is None as it is already supplied
# in the dataset.
clf.fit(X_train, y=y_train)


print("Model trained.")


# -------- EVALUACIÓN --
print("Evaluating model...")
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


preds = clf.predict(X_test)

acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average="micro")
precision = precision_score(y_test, preds, average="micro")
recall = recall_score(y_test, preds, average="micro")
print(f"Accuracy: {acc:.3f}")
print(f"F1 Score: {f1:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall: {recall:.3f}")
cm = confusion_matrix(y_test, preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title(MODEL_NAME)

disp.figure_.savefig(f"Braindecode/CM_AttentionBaseNet.png")

print("Model evaluated.")