import glob
import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
print("tf.__version__", tf.__version__)

import load
import trans
import config 
import learn
import inference
import neptune.new as neptune
from neptune.new.integrations.tensorflow_keras import NeptuneCallback

run = neptune.init(
    project="signlanguages/ml-epenthesis",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJkMjIyMDAxYy00OWI1LTQ2YWItODg2ZS03ZGJmNDllNGE4NjkifQ==",
    mode=config.OFFLINE_ASYNC
    #run=""
)  


# Check and determine сount of the files of skelet models and weakly labeled movements
PATH_FOLDER_CSV = "/home/cv2020/YAYAY/GitHub/slsru_ml_tag/data/csv_slsru_skelet_v0_1_0/x_x_cx_and_sxx_interpol_30FPS/"
PATH_FOLDER_JSON = "/home/cv2020/YAYAY/GitHub/slsru_ml_tag/data/json/WeaklyLabeledMovement/x_x_cx_and_sxx_interpol_30FPS/"
list_csv = glob.glob(PATH_FOLDER_CSV + "*")
list_json = glob.glob(PATH_FOLDER_JSON + "*")
print("len(list_csv)", len(list_csv))
print("len(list_json)", len(list_json))

run["data/x_x_cx_and_sxx_interpol_30FPS"].track_files(PATH_FOLDER_CSV)
run["label/x_x_cx_and_sxx_interpol_30FPS"].track_files(PATH_FOLDER_JSON)

params = {
        "lr": config.LEARNING_RATE, 
        "epochs": config.EPOCHS, 
        "batch_size": config.BATCH_SIZE, 
        "window_size":config.WINDOW_SIZE,
        "load_fast": config.MODE,
        "select_features": config.SELECT_FEATURES,
        "train_val_test": str(1 - config.TEST_SIZE) + "_" + str(config.TEST_SIZE) + "_" + str(config.TEST_SIZE),
        "trans": config.TRANS_FEATURES,
        "inbalance2balance": config.INBALANCE2BALANCE,
        }
run["param"] = params
run["model_net"] = config.MODEL_NET

neptune_cbk = NeptuneCallback(run=run, base_namespace="fit")

# Load from csv and json files to general merging training data
# If rename paramer mode="fast" to mode="" then will load from all files wait a few minute else fast load from one file only
x_full, y_full = load.from_json_csv_by_folders(PATH_FOLDER_CSV, PATH_FOLDER_JSON, mode=config.MODE)

# Convert coordinate points of skelet model to angles between points
if config.TRANS_FEATURES:
    x_full = trans.tranform_angles(x_full)

# Convert general merging training data to windowing data on frame rate series data
if config.WINDOW_SIZE:
    x_steps, y_steps = trans.windowing_xy(x_full, y_full, config.WINDOW_SIZE, y_last_or_center=config.CENTER_LAST)




# Check imbalancing classes

import seaborn as sns
import pandas as pd
df_y_full = pd.DataFrame(data=y_steps, columns=["class"])
# Only on jupyter
sns.countplot(x="class", data=df_y_full )


## Reduce training data to balancing classes
if config.INBALANCE2BALANCE:
    x_steps, y_steps = trans.inbalance2balance_postwindow(x_steps,y_steps)
    df_y_steps = pd.DataFrame(data=y_steps, columns=["class"])
    # Only on jupyter
    sns.countplot(x="class", data=df_y_steps )

# Split training and validation data
X_train, X_val, X_test, Y_train, Y_val, Y_test = learn.split_train_val_test(x_steps, y_steps, config.TEST_SIZE)

# Y to categotical

Y_train_one = tf.keras.utils.to_categorical(Y_train)
Y_val_one = tf.keras.utils.to_categorical(Y_val)
Y_test_one = tf.keras.utils.to_categorical(Y_test)

# Learning
model = learn.model_fn(X_val.shape, Y_val_one.shape)

history = model.fit(X_train, Y_train_one, epochs=config.EPOCHS, verbose=2,
        batch_size=config.BATCH_SIZE, validation_data = (X_val, Y_val_one),callbacks=[neptune_cbk])

# Save model
name_model = "test"
path_model_write = "../model/" + name_model
tf.saved_model.save(model, path_model_write + "_tf")
model.save(path_model_write + "_keras.h5")
model.save(path_model_write +"_tfv2", save_format="tf")

# Predict
Y_pred_one = model.predict(X_test, verbose=2)
print("Y_pred_one[0:10]", Y_pred_one[0:10])
Y_pred = []
for p in Y_pred_one:
  #Y_pred.append(np.round(p[0]))
  Y_pred.append(np.argmax(p))
print("len(Y_pred)", len(Y_pred))
print("Y_pred[0:10]", Y_pred)
print("Y_test[0:10]", Y_test.tolist())

# Evalution model
eval_metrics = model.evaluate(X_test, Y_test_one)
print("test/loss", eval_metrics[0])
print("test/acc", eval_metrics[1])

# Confusion matrix
"""
Follow classes:
0 - Idle
1 - Begin movement
2 - Transitional movement(epenthesis)
3 - End movement
4 - Sign(gloss)
"""
cnf_matrix = confusion_matrix(Y_test, Y_pred, labels=[0,1,2,3,4])

df_cnf_matrix = inference.df_confusion_matrix(cnf_matrix, ["idle","start","trans","end","sign"])
print("cnf_matrix/test_normaliz", df_cnf_matrix)
run['cnf_matrix/test_normaliz'].upload(neptune.types.File.as_html(df_cnf_matrix))

df_cnf_matrix = inference.df_confusion_matrix(cnf_matrix, ["idle","start","trans","end","sign"], normalize=False)
print("cnf_matrix/test", df_cnf_matrix)
run['cnf_matrix/test'].upload(neptune.types.File.as_html(df_cnf_matrix))
