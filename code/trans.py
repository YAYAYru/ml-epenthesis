import numpy as np
from collections import Counter
from numpy.lib.arraysetops import unique
#from skelet_mediapipe import *
import pandas as pd
import glob
import matplotlib.pyplot as plt
import json

import config
import load

"""
def windowing_xy(x_full, y_full, step):
    step_x = []
    step_y = []

    for i in range(x_full.shape[0]-step):
        step_x.append(x_full[i:(i+step)])
        y_mean = np.median(y_full[i:(i+step)])
        str_y = str(y_mean)
        if int(str_y.split(".")[1]) >= 5:
            y_mean = int(str_y.split(".")[0]) + 1 
        else:
            y_mean = int(str_y.split(".")[0])
        step_y.append(y_mean)
    step_x = np.array(step_x)
    step_y = np.array(step_y)
    print("step_x.shape", step_x.shape)
    print("step_y.shape", step_y.shape)

    print("np.unique(step_y", np.unique(step_y))
    print("Counter(y_full)",Counter(y_full))

    return step_x, step_y
"""

def find_angle_xyz(a, b, b1, c):
    #a = np.array([32.49, -39.96,-3.86])
    #b = np.array([31.39, -39.28, -4.66])
    #c = np.array([31.14, -38.09,-4.49])
    #print("a, b, b1, c", a)
    ba = a - b
    bc = c - b1
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    angle = angle/3.141592
    return angle    

def tranform_angle(feature):
    #https://www.omnicalculator.com/math/angle-between-two-vectors
    if config.TRANS_FEATURES=="angle_14_12_11_13":
        pose_points = {
            "right_elbow":[16, 14, 14, 12], 
            "right_shoudler":[14, 12, 12, 11],
            "left_shoudler":[12, 11, 11, 13], 
            "left_elbow":[11, 13, 13, 15]     
        }
        x = feature[0::2]
        y = feature[1::2]
    angles = []
    sels = [n for n in pose_points.values()]
    #print("sels", sels)
    
    for n in sels:
        a, b, b1, c = [x[n[0]], y[n[0]]], [x[n[1]], y[n[1]]],[x[n[2]], y[n[2]]], [x[n[3]], y[n[3]]]
        #print("a, b, b1, c", a, b, b1, c)
        a = np.array(a)
        b = np.array(b)
        b1 = np.array(b1)
        c = np.array(c)
        angles.append(find_angle_xyz(a, b, b1, c))

    X = np.array(angles)
    #X = X / np.array([180])
    #angle = find_angle_xyz(n1, n2, n3, n4)
    #print("angle", angle)
    return X
def tranform_angles(features):
    feature_angles = []
    for n in features:
        feature_angles.append(tranform_angle(n))
    a = np.array(feature_angles)
    print("tranform_angles(features).shape", a.shape)
    return a 


def test_tranform_angle(feature):
    # тест показания минимальные и максимлаьные углы
    min_c=[]
    max_c=[]
    for n in feature:
        coords_angles = tranform_angle(n)    
    min_c.append(coords_angles.min())
    max_c.append(coords_angles.max())
    print("np.array(min_c).min()", np.array(min_c).min())
    print("np.array(max_c).max()", np.array(max_c).max())

def inbalance2balance_df(df_XY,labelname):
    # Human Activity Recognition Using Accelerometer Data
    # https://github.com/laxmimerit/Human-Activity-Recognition-Using-Accelerometer-Data-and-CNN
    print("XY", df_XY[labelname].value_counts())
    min = df_XY[labelname].value_counts().min()
    idle0 = df_XY[df_XY[labelname]==0].head(min).copy()
    start1 = df_XY[df_XY[labelname]==1].head(min).copy()
    trans2 = df_XY[df_XY[labelname]==2].head(min).copy()
    end3 = df_XY[df_XY[labelname]==3].head(min).copy()
    sign4 = df_XY[df_XY[labelname]==4].copy()
    balanced_data = pd.DataFrame()
    balanced_data = balanced_data.append([idle0, start1, trans2, end3, sign4])
    print("balanced_data['label'].value_counts()", balanced_data[labelname].value_counts())
    return balanced_data

def inbalance2balance_np(x_full_angle,y_full):
    df=pd.DataFrame()
    if config.TRANS_FEATURES=="angle_14_12_11_13":
        df = pd.DataFrame({"label":y_full,"a14":x_full_angle[:,0],"a12":x_full_angle[:,1],"a11":x_full_angle[:,2],"a13":x_full_angle[:,3]})
        df = inbalance2balance_df(df,"label")
    Y = df["label"].to_numpy()
    X = df.to_numpy()[:,1:]
    return X,Y
def tranform_angles_df2df(df):
    np_a = tranform_angles(df.to_numpy())
    if config.TRANS_FEATURES=="angle_14_12_11_13":
        column_names = {
            "right_elbow",
            "right_shoudler",
            "left_shoudler", 
            "left_elbow"}
    return pd.DataFrame(np_a, columns=column_names)

def stat_fps(csvlist):
    fps_list = []
    d = Counter()
    for n in csvlist:
        _,fps = load.from_csv_slsru_skelet_v0_1_0_df(n, config.SELECT_FEATURES,fps=True)
        fps = round(fps)
        d[str(fps)] +=1
        #fps_list.append(fps)
    path = "../data/fps/stat_fps.csv"
    df = pd.DataFrame.from_dict(d, orient='index')
    
    df.to_csv(path)
    print("df", df)


def windowing_xy(x_full, y_full, step, x_to_y="ManyToOne", y_last_or_center="last"):
    step_x = []
    step_y = []

    if x_to_y=="ManyToOne":
        if y_last_or_center=="last":
            for i in range(x_full.shape[0]-step):
                step_x.append(x_full[i:(i+step)])
                y_mean = np.median(y_full[i:(i+step)])
                str_y = str(y_mean)
                if int(str_y.split(".")[1]) >= 5:
                    y_mean = int(str_y.split(".")[0]) + 1 
                else:
                    y_mean = int(str_y.split(".")[0])
                step_y.append(y_mean)
        if y_last_or_center=="center":
            #for i in range(x_full.shape[0]-step):# Была проблема с того, что после виндоуса один элемент пропускается
            for i in range(x_full.shape[0]-step+1):
                step_x.append(x_full[i:(i+step)])
                y_mean = y_full[i+int(step/2)]
                step_y.append(y_mean)

    if x_to_y=="ManyToMany":
        for i in range(x_full.shape[0]-step):
            step_x.append(x_full[i:(i+step)])
            step_y.append(y_full[i:(i+step)])        

    step_x = np.array(step_x)
    step_y = np.array(step_y)
    print("step_x.shape", step_x.shape)
    print("step_y.shape", step_y.shape)

    print("np.unique(step_y", np.unique(step_y))
    print("Counter(y_full)",Counter(y_full))

    return step_x, step_y

def windowing_x(x_full, step, x_to_y="ManyToOne", y_last_or_center="last"):
    step_x = []
    step_y = []

    if x_to_y=="ManyToOne":
        if y_last_or_center=="last":
            for i in range(x_full.shape[0]-step):
                step_x.append(x_full[i:(i+step)])
                #y_mean = np.median(y_full[i:(i+step)])
                str_y = str(y_mean)
                if int(str_y.split(".")[1]) >= 5:
                    y_mean = int(str_y.split(".")[0]) + 1 
                else:
                    y_mean = int(str_y.split(".")[0])
                #step_y.append(y_mean)
        if y_last_or_center=="center":
            for i in range(x_full.shape[0]-step+1): #была проблема с тем, что после виндоуса пропускается один элемент
            #for i in range(x_full.shape[0]-step):
                step_x.append(x_full[i:(i+step)])
                #y_mean = y_full[i+int(step/2)]
                #step_y.append(y_mean)

    if x_to_y=="ManyToMany":
        for i in range(x_full.shape[0]-step):
            step_x.append(x_full[i:(i+step)])
            #step_y.append(y_full[i:(i+step)])        

    step_x = np.array(step_x)
    #step_y = np.array(step_y)
    print("step_x.shape", step_x.shape)
    #print("step_y.shape", step_y.shape)

    #print("np.unique(step_y", np.unique(step_y))
    #print("Counter(y_full)",Counter(y_full))

    #return step_x, step_y
    return step_x

def inbalance2balance_postwindow(x, y):
    print("x", np.array(x).shape)
    print("y", np.array(y).shape)
    print("Counter(y)",Counter(y))
    counter = Counter(y)
    min_key = min(counter, key=counter.get)
    print("min_key", min_key)
    print("min_value", counter[min_key])

    X = []
    Y = []
    j = 0
    labelnp= np.unique(y)
    print("labelnp", labelnp)

    for n in labelnp:
        for i,n1 in enumerate(y):
            if n == n1 and counter[min_key]>j:
                X.append(x[i])
                Y.append(y[i])
                j = j + 1
        j = 0

    counter_Y = Counter(Y)
    print("counter_Y", counter_Y)
    X = np.array(X)
    Y = np.array(Y)

    return X, Y

if __name__ == '__main__':

    x_full, y_full = load.from_json_csv_by_folders(config.PATH_CSV_SKELET, config.PATH_JSON_WLM, mode=config.MODE)
    x_full_angle = tranform_angles(x_full)
    print("x_full_angle", x_full_angle)
    print("y_full", y_full)
    y_full = load.multiclass2binaryclass(y_full)
    x_steps, y_steps = windowing_xy(x_full, y_full, config.WINDOW_SIZE, y_last_or_center="center")
    #print("x_steps", x_steps)
    #print("y_steps", y_steps)

    X,Y = inbalance2balance_postwindow(x_steps, y_steps)
    #print("X", X)
    #print("Y", Y)
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)

    #s = 100
    #e = 105
    #x_full, y_full = load.from_json_csv_by_folders(config.PATH_CSV_SKELET, config.PATH_JSON_WLM, mode=config.MODE)
    #x_full = tranform_angles(x_full)    

    """
    x_full = np.array(range(14))
    #y_full = np.array([1,1,1,2,2,2,4,4,4,4,4,4,3,0])
    y_full = np.array(range(14))
    print("x_full", x_full)
    print("y_full", y_full)
    print("x_full.shape", x_full.shape)
    print("y_full.shape", y_full.shape)
    xy_full = [[x_full[i], y_full[i]] for i ,_ in enumerate(x_full)]
    print("xy_full", xy_full)
    print("---------------windowing_xy()----------------")
    x_steps, y_steps = windowing_xy(x_full, y_full, 7, x_to_y="ManyToOne", y_last_or_center="center") 
    #print("x_steps[:n] ManyToOne", x_steps[s:e])
    #print("x_steps[:n] ManyToOne", y_steps[s:e])
    print("x_steps", x_steps)
    print("y_steps", y_steps)
    print("x_steps.shape", x_steps.shape)
    print("y_steps.shape", y_steps.shape)
    """

    """
    csvlist = glob.glob(config.PATH_CSV_SKELET_X_X_CX + '*.csv')
    jsonlist = glob.glob(config.PATH_JSON_WLM_X_X_CX + '*.json')
    videolist = load.set_csv_list(csvlist,jsonlist)     
    df_data, fps, df_conf = load.from_csv_slsru_skelet_v0_1_0_df(config.PATH_CSV_SKELET_X_X_CX+videolist[0]+".csv", config.SELECT_FEATURES,fps=True,conf=True)
    np_label = load.from_json_supervisely_WLM(config.PATH_JSON_WLM_X_X_CX+videolist[0]+".json")
    #visual.plot_angles(df_data, np_label)
    df_data, np_label = interpolation_XY_label(df_data,df_conf,np_label,fps,to_fps=30)
    #visual.plot_angles(df_data, np_label)
    """
    
    """
    videoname = "176632_4_3.mp4"
    df_data, fps, df_conf = load.from_csv_slsru_skelet_v0_1_0_df(config.PATH_CSV_SKELET_X_X_CX+videoname+".csv", config.SELECT_FEATURES,fps=True,conf=True)
    np_label = load.from_json_supervisely_WLM(config.PATH_JSON_WLM_X_X_CX+videoname+".json")
    #visual.plot_angles(df_data, np_label)
    print("np_label", np_label)
    df_data, np_label = interpolation_XY_label(df_data,df_conf,np_label,fps,to_fps=30)
    print("np_label", np_label)
    #visual.plot_angles(df_data, np_label)
    """
    # videoname = "176632_4_3.mp4"
    """
    videoname = "30_7_c3.mp4"
    path_csv = config.PATH_CSV_SKELET_X_X_CX+videoname+".csv"
    path_json = config.PATH_JSON_WLM_X_X_CX+videoname+".json"
    to_path_csv  = "../data/csv_slsru_skelet_v0_1_0/x_x_cx_interpol_30FPS_temp/"+videoname+".csv"
    to_path_json = "../data/json/WeaklyLabeledMovement/x_x_cx_interpol_30FPS_temp/"+videoname+".json"
    interpolation_XY_label_file2file(30, path_csv, path_json, to_path_csv, to_path_json)  
    """
    
    """
    print("-------interpolation_XY_label_folder2folder------------")
    to_folder_csv = "../data/csv_slsru_skelet_v0_1_0/x_x_cx_interpol_30FPS/"
    to_folder_json = "../data/json/WeaklyLabeledMovement/x_x_cx_interpol_30FPS/"
    interpolation_XY_label_folder2folder(30, config.PATH_CSV_SKELET_X_X_CX, config.PATH_JSON_WLM_X_X_CX, to_folder_csv, to_folder_json)
    """

    """
    print("-------interpolation_XY_label_folder2folder------------")
    PATH_JSON_WLM = "../data/json/WeaklyLabeledMovement/sxx_9and11_c5_trans/"
    PATH_CSV_SKELET = "../data/csv_slsru_skelet_v0_1_0/sxx_9and11_c5_trans/"
    to_folder_csv = "../data/csv_slsru_skelet_v0_1_0/sxx_9and11_c5_trans_interpol_30FPS/"
    to_folder_json = "../data/json/WeaklyLabeledMovement/sxx_9and11_c5_trans_interpol_30FPS/"
    interpolation_XY_label_folder2folder(30, PATH_CSV_SKELET, PATH_JSON_WLM, to_folder_csv, to_folder_json)
    """
    
    #print("-------load.from_json_csv_by_folders---------")
    #x_full, y_full = load.from_json_csv_by_folders(to_folder_csv, to_folder_json, mode=config.MODE)
    #print("x_full", x_full)
    #print("y_full", y_full)

    #print("x_full.shape", x_full.shape)
    #print("y_full.shape", y_full.shape)



    """
    videofile = "295184_9_c5.mp4"
    df_data, fps, df_conf = load.from_csv_slsru_skelet_v0_1_0_df(config.PATH_CSV_SKELET_X_X_CX+videofile+".csv", config.SELECT_FEATURES,fps=True,conf=True)
    np_label = load.from_json_supervisely_WLM(config.PATH_JSON_WLM_X_X_CX+videofile+".json")
    """

    
    """
    s = 150
    e = 165
    x_full, y_full = load.from_json_csv_by_folders(config.PATH_CSV_SKELET, config.PATH_JSON_WLM, mode=config.MODE)
    x_full = tranform_angles(x_full)    

    print("---------------windowing_xy()----------------")
    x_steps, y_steps = windowing_xy(x_full, y_full, config.WINDOW_SIZE, x_to_y="ManyToOne") 
    print("x_steps[:n] ManyToOne", x_steps[s:e])
    print("x_steps[:n] ManyToOne", y_steps[s:e])

    x_steps, y_steps = windowing_xy(x_full, y_full, config.WINDOW_SIZE, x_to_y="ManyToMany") 
    print("x_steps[:n] ManyToMany", x_steps[s:e])
    print("x_steps[:n] ManyToMany", y_steps[s:e])
    """


    """
    path_csv = "../data/csv_slsru_skelet_v0_1_0/sl_sentence_DianaB_DimaG/ss1_9_c5_1.mp4.csv"
    features = load.from_csv_slsru_skelet_v0_1_0(path_csv, config.SELECT_FEATURES)
    test_tranform_angle(features)
    print("features.shape", features.shape)
    features = tranform_angles(features)
    print("features.shape", features.shape)  
    print("features", features) 
    """  

    # path_video = "../data/video/burkova1006/x_5_x/63_5_1.mp4"
    # video2skelet(path_video)
    """
    x_full, y_full = load.from_json_csv_by_folders(config.PATH_CSV_SKELET, config.PATH_JSON_WLM, mode=config.MODE,)
    x_full_angle = tranform_angles(x_full)
    print("x_full_angle", x_full_angle)
    print("y_full", y_full)
    X,Y = inbalance2balance_np(x_full_angle, y_full)
    print("Y", Y.shape)
    print("Y", Y)
    print("X", X.shape)
    print("X", X)
    """


    # stat_fps(csvlist)    
    pass
