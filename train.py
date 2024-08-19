### Library and file imports
import glob
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    recall_score,
    precision_score,
    f1_score,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, FunctionTransformer
from cloudpickle import dump

path_to_normal_operation_files = r"./data/0"
path_to_flow_instability_files = r"./data/4"
all_normal_operation_files = glob.glob(
    os.path.join(path_to_normal_operation_files, "*.csv")
)
all_flow_instability_files = glob.glob(
    os.path.join(path_to_flow_instability_files, "*.csv")
)

print("Reading files...")
### Base samples
list_files = []
for filename in all_normal_operation_files:
    df_normal = pd.read_csv(filename, index_col=None, header=0)
    list_files.append(df_normal)

normal_frame = pd.concat(list_files, axis=0)
del list_files, df_normal
normal_frame.head(3)

list_files = []
for filename in all_flow_instability_files:
    df_anomaly = pd.read_csv(filename, index_col=None, header=0)
    list_files.append(df_anomaly)

flow_instability_frame = pd.concat(list_files, axis=0)
del list_files, df_anomaly
flow_instability_frame.head(3)

df_source = pd.concat([normal_frame, flow_instability_frame])
del normal_frame, flow_instability_frame
print("All files read!")

print("Starting preprocessing...")
### Creating Pipeline
## Preprocessing of the whole base
# Observations without classification are discarded
target_class = "class"
df_pipe = df_source.dropna(subset=[target_class])
X_pipe = df_pipe.drop(target_class, axis=1)
y_pipe = df_pipe.loc[:, target_class]
del df_pipe

X_pipe_train, X_pipe_test, y_pipe_train, y_pipe_test = train_test_split(
    X_pipe, y_pipe, test_size=0.2, random_state=666
)
del X_pipe, y_pipe
print("Preprocessing done!")

print("Assembling and executing pipeline...")
## Assembling pipeline with defined steps
columns_to_drop = ["timestamp", "T-JUS-CKGL"]

drop_function = FunctionTransformer(lambda x: x.drop(columns=columns_to_drop, axis=1))
fill_mean = SimpleImputer(missing_values=np.nan, strategy="mean")
pipe = Pipeline(
    steps=[
        ("dropout", drop_function),
        ("fillna", fill_mean),
        ("transform", MinMaxScaler()),
        ("model", GaussianNB()),
    ]
)
pipe.fit(X_pipe_train, y_pipe_train)
print("Pipeline executed!")

print("Creating metrics and report...")
## Metrics
predictions = pipe.predict(X_pipe_test)
metrics_data = {
    "accuracy": (pipe.score(X_pipe_test, y_pipe_test)),
    "recall": (recall_score(y_pipe_test.values, predictions, pos_label=4)),
    "precision": (precision_score(y_pipe_test.values, predictions, pos_label=4)),
    "f1-score": (f1_score(y_pipe_test.values, predictions, pos_label=4)),
    "cm": confusion_matrix(y_pipe_test, predictions),
}

try:
    os.mkdir("results")
except FileExistsError:
    pass

with open("results/metrics.txt", "w") as outfile:
    outfile.write(
        f"\nAccuracy = {round(metrics_data['accuracy'], 4)}, "
        + f"Recall = {round(metrics_data['recall'], 4)}, "
        + f"Precision = {round(metrics_data['precision'], 4)}, "
        + f"F1 Score = {round(metrics_data['f1-score'], 4)}"
    )

## Results Report
sns.set_theme()
fig, axs = plt.subplots(1, 2, figsize=(10, 3))

sns.heatmap(
    metrics_data["cm"] / np.sum(metrics_data["cm"]),
    annot=True,
    fmt=".2%",
    cmap="Blues",
    ax=axs[0],
)
axs[0].set_xlabel("Predicted Labels")
axs[0].set_ylabel("True Labels")
axs[0].xaxis.set_ticklabels(["normal", "anomaly"])
axs[0].yaxis.set_ticklabels(["normal", "anomaly"])
axs[0].set_title("Confusion Matrix")

fpr, tpr, thresholds = roc_curve(y_pipe_test, predictions, pos_label=4)
auc_score = auc(fpr, tpr)
axs[1].plot(fpr, tpr, label=f"(AUC = {auc_score:.2f})", color="red")
axs[1].plot([0, 1], [0, 1], color="black", linestyle="--")
axs[1].set_xlim([0.0, 1.0])
axs[1].set_ylim([0.0, 1.05])
axs[1].set_xlabel("False Positive Rate")
axs[1].set_ylabel("True Positive Rate")
axs[1].set_title(f"ROC Curve")
axs[1].legend(loc="lower right")

plt.plot()
plt.savefig("results/model_results.png", dpi=120)

## Saving Pipeline
model_file = "model/anomaly_detector_pipeline.pkl"

try:
    os.mkdir("model")
except FileExistsError:
    pass

with open(model_file, "wb") as pkl_file:
    dump(pipe, pkl_file)

print("Done!")
print("Training executed with success.")
