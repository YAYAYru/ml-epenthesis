# Path
#RUN_ID = "SLSMLTAG-109"
#RUN_ID = "SLSRUM2CLA-95"
RUN_ID = "SLSRUM2CLA-123"

# PATH_JSON_WLM = "../data/json/WeaklyLabeledMovement/sxx_9and11_c5_trans/"
# PATH_CSV_SKELET = "../data/csv_slsru_skelet_v0_1_0/sl_sentence_DianaB_DimaG/"
PATH_JSON_WLM = "../data/json/WeaklyLabeledMovement/x_x_cx_30FPS/"
PATH_CSV_SKELET = "../data/csv_slsru_skelet_v0_1_0/x_x_cx_30FPS/"
PATH_MODEL = "../model/neptune/" + RUN_ID
PATH_IMAGE_HISTORY = "../image/hystory.png"
PATH_IMAGE_CONFUSION_MATRIX = "../image/confusion_matrix_test.png"

PATH_JSON_WLM_X_X_CX = "../data/json/WeaklyLabeledMovement/x_x_cx/"
PATH_CSV_SKELET_X_X_CX = "../data/csv_slsru_skelet_v0_1_0/x_x_cx/"

PATH_JSON_WLM_X_X_CX_INTERPOL_30FPS = "../data/json/WeaklyLabeledMovement/x_x_cx_interpol_30FPS/"
PATH_CSV_SKELET_X_X_CX_INTERPOL_30FPS = "../data/csv_slsru_skelet_v0_1_0/x_x_cx_interpol_30FPS/"

PATH_JSON_WLM_X_X_CX_SXX_INTERPOL_30FPS = "../data/json/WeaklyLabeledMovement/x_x_cx_and_sxx_interpol_30FPS/"
PATH_CSV_SKELET_X_X_CX_SXX_INTERPOL_30FPS = "../data/csv_slsru_skelet_v0_1_0/x_x_cx_and_sxx_interpol_30FPS/"



# Load
# - from_json_csv_by_folders(mode="fast")
MODE = ""
OFFLINE_ASYNC="async"
#https://google.github.io/mediapipe/solutions/pose.html
# "pose_xy_25" or "pose_xyz_25" or "pose_xy_33" or "pose_xyz_33"
SELECT_FEATURES = "pose_xy_25"
TRANS_FEATURES = "angle_14_12_11_13"
INBALANCE2BALANCE = 0
INBALANCE2BALANCE_POSTWINDOW = 1
AUG_STEP = ""

# Trans
WINDOW_SIZE = 31

#Learn
#MODEL_NET = "model_fn_v1_timedistributed"
#MODEL_NET = "model_many_to_many_2class_v1"
#MODEL_NET = "model_many_to_one_2class_v3"
MODEL_NET = "model_fn_v1"
EPOCHS = 1000
BATCH_SIZE = 1000
TEST_SIZE = 0.1
LEARNING_RATE = 0.0001
