# Path
#RUN_ID = "MLEP-5"

# Load
# - from_json_csv_by_folders(mode="fast")
MODE = "fast"
OFFLINE_ASYNC="async"
#https://google.github.io/mediapipe/solutions/pose.html
# "pose_xy_25" or "pose_xyz_25" or "pose_xy_33" or "pose_xyz_33"
SELECT_FEATURES = "pose_xy_25"
TRANS_FEATURES = "angle_14_12_11_13"
INBALANCE2BALANCE = 1

# Trans
WINDOW_SIZE = 31
CENTER_LAST = "center"
MODEL_NET = "model_fn_v1" # in learn.py
EPOCHS = 10
BATCH_SIZE = 1000
TEST_SIZE = 0.1
LEARNING_RATE = 0.0001
