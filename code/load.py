import json
import glob
import numpy as np
import os
import pandas as pd
import datetime
from shutil import copyfile
import os

import config


#np.set_printoptions(threshold=sys.maxsize)
#TODO Visualization from json and csv

def from_json_supervisely_WLM(path_json):
    """    
    Load from supervisely json by Weakly Labeled Movement
    :param path_json: path to folder where json files
    :return: 0 - idle, 1 - begin, 2 - trans, 3 - end, 4 - sign
    :example:
        filelist = glob.glob(path_json + '*.json')
        y = load_json_supervisely_WLM(filelist[5])
        print(y)
    """
    with open(path_json) as test:
        data = json.load(test)
    
    #y = np.zeros(data["framesCount"], dtype=int)
    begin = []
    end = []
    middle =[]

    for t in data["tags"]:
        if t["name"] == "BeginMovement":
            begin.append(t["frameRange"])
        if t["name"] == "EndMovement":
            end.append(t["frameRange"])
        if t["name"] == "TransitionalMovement":
            middle.append(t["frameRange"])   

    begin = sorted(begin)
    middle = sorted(middle)
    end = sorted(end)

    y = np.zeros(data["framesCount"], dtype="int")
    # Если появится ошибка здесь, то нужно тестировать фукнцией wlm_json_validation(path_json)
    
    for i, _ in enumerate(begin):
        y[begin[i][1]:end[i][0]] = 4

    for i in begin:
        y[i[0]:i[1]] = 1
    for i in middle:
        y[i[0]:i[1]] = 2
    for i in end:
        y[i[0]:i[1]] = 3

    return y


        
    """
    #y = np.zeros(data["framesCount"], dtype="str")
    y = np.full(data["framesCount"], "idle")
    for i,n in enumerate(begin):
        y[begin[i][1]:end[i][0]] = 'sign'
    for i in begin:
        y[i[0]:i[1]] = 'begin'
    for i in middle:
        y[i[0]:i[1]] = 'trans'
    for i in end:
        y[i[0]:i[1]] = 'end'
    return y    
 
    """
def from_csv_slsru_skelet_v0_1_0_df(path_csv, select_features, fps=False, conf=False):
    """
    Load from csv by mediapipe and mediapipe_video2skelet_csv.py
    :param path_json: path to folder where csv files
    :param select_features: 0 - all, 1 - pose_xy, 2 - pose_xyz
    :return:
        csvlist = glob.glob(path_csv + '/*/*.csv')
        x = load_csv_slsru_skelet_v0_1_0(csvlist[5])  
        print(x)  
    """
    df = pd.read_csv(path_csv)
    #first_feature = 4
    #df = df.iloc[:,range(first_feature, first_feature+33*4)]

    feature_names = []
    feature_confidence = []
    if select_features=="pose_xy_33":
        for i in range(33):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_confidence.append("pose_p" + str(i))
    if select_features=="pose_xyz_33":
        for i in range(33):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_names.append("pose_z" + str(i))
            feature_confidence.append("pose_p" + str(i))
    if select_features=="pose_xy_25":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_confidence.append("pose_p" + str(i))
    if select_features=="pose_xyz_25":
        for i in range(25):
            feature_names.append("pose_x" + str(i))
            feature_names.append("pose_y" + str(i))
            feature_names.append("pose_z" + str(i))
            feature_confidence.append("pose_p" + str(i))
    if len(feature_names)==0:
        x=df
    else:
        x=df[feature_names]
    if fps==True:
        return x, df["fps"][0], df[feature_confidence]
    return x

def from_csv_slsru_skelet_v0_1_0(path_csv, select_features):
    x = from_csv_slsru_skelet_v0_1_0_df(path_csv, select_features)
    return x.to_numpy()

def from_json_csv_by_video(path_csv, path_json, videoname):
    """
    :param path_csv: path to folder where csv files
    :param path_json: path to folder where json files
    :videoname: video name
    :example:
        videoname = "63_5_1.mp4"
        path_json = "../data/json/WeaklyLabeledMovement_v2_json/4and5and6/ann/"
        path_csv = "../data/csv_slsru_skelet_v0_1_0/burkova1006/x_5_x/"
        x, y = from_json_csv_by_video(path_csv, path_json, videoname)
        print(x.shape)
        print(y.shape)
    """
    x = from_csv_slsru_skelet_v0_1_0(path_csv+videoname+'.csv', select_features=config.SELECT_FEATURES)
    y = from_json_supervisely_WLM(path_json+videoname+'.json')
    return x, y

"""
def numpyXY_to_csv(x,y):
    print("x.shape", x.shape)
    print("y.shape", y.shape)
"""


def from_json_csv_by_folders(path_folder_from_csv, path_folder_from_json, mode="fast", path_csv=""):
    """
    :param path_folder_csv: path to folder where csv files
    :param path_folder_json: path to folder where json files
    :param mode: "fast" - learn only 5 exampes; "" - slow learn all examples 
    :example:
        path_json = "../data/json/WeaklyLabeledMovement_v2_json/4and5and6/ann/"
        path_csv = "../data/csv_slsru_skelet_v0_1_0/burkova1006/x_5_x/"
        X, Y = from_json_csv_by_folders(path_csv, path_json)
    """

    #x = path_folder_from_csv
    #y = path_folder_from_json
    print("-- from_json_csv_by_folders()")

    pathcsv_list = glob.glob(path_folder_from_csv + '/**/*.csv', recursive=True)
    len_csv = len(pathcsv_list)
    print('len(pathcsv_list):', len_csv)
    
    pathjson_list = glob.glob(path_folder_from_json + '/**/*.json', recursive=True)
    len_json = len(pathjson_list)
    print('len(pathjson_list):', len_json)

    videolist = set_csv_list(pathcsv_list, pathjson_list)

    X, Y = from_json_csv_by_video(path_folder_from_csv, path_folder_from_json, videolist[0])
    print('videolist.shape', np.array(videolist).shape)
    if mode == "fast":
        for n in videolist[:1]:
            print("n", n)
            x, y = from_json_csv_by_video(path_folder_from_csv, path_folder_from_json, n)
            X = np.append(X,x, axis=0)
            Y = np.append(Y,y, axis=0)

    else:
        for n in videolist:
            x, y = from_json_csv_by_video(path_folder_from_csv, path_folder_from_json, n)
            X = np.append(X,x, axis=0)
            Y = np.append(Y,y, axis=0)      

    # Разные числа , потому что разные файлы в связи с glob  
    print("X.shape", X.shape)
    print("Y.shape", Y.shape)
    
    if path_csv!="":
        col_header = []
        i=0
        while i < X.shape[1]/2:
            col_header.append("pose_x" + str(i))
            col_header.append("pose_y" + str(i))
            i=i+1

        df = pd.DataFrame(X, columns=col_header)
        df["type_movement"] = Y.tolist()
        df.to_csv(path_csv, index=False)
        print("Written to csv")
    return X,Y

def wlm_jsons_validation(path_folder):
    filelist = glob.glob(path_folder)
    #filelist = ["../data/json/WeaklyLabeledMovement/x_x_cx/789_3_2.mp4.json", "../data/json/WeaklyLabeledMovement/x_x_cx/1000254_3_2.mp4.json"]
    #print(filelist)
    c = 0
    for n in filelist:
        if wlm_json_validation(n)==False:
            c=c+1
    print("count -", c)

def wlm_json_validation(path_json):
    with open(path_json) as test:
        data = json.load(test)
    begin = []
    end = []
    middle =[]
    #print("data[tags]", len(data["tags"]))
    if len(data["tags"])==0:
        print("Empty tags")
        return False 

    for t in data["tags"]:
        if t["name"] == "BeginMovement":
            begin.append(t["frameRange"])
        if t["name"] == "EndMovement":
            end.append(t["frameRange"])
        if t["name"] == "TransitionalMovement":
            middle.append(t["frameRange"])   

    
    if np.array(begin).shape == np.array(end).shape:
        pass
    else:
        print("begin.shape and end.shape do not match")
        print("path", path_json)
        print("begin.shape", np.array(begin).shape)
        print("end.shape", np.array(end).shape) 
        return False  

    return True


import glob
import os
import shutil
def copy_filecsvs_by(path_folder, from_foldercsv, to_foldercsv, log=False):
    path_jsons = glob.glob(path_folder + "*.json")
    
    filecsvs = []
    for n in path_jsons:
        n = os.path.normpath(n)
        n = n.split(os.sep)[-1]
        n = n.replace(".json", ".csv")
        filecsvs.append(n)
    print("len(path_files) - json", len(path_jsons))

    count_copy = 0
    if log:
        for i, n in enumerate(filecsvs):
            path_csv = from_foldercsv + "/" + n
            if os.path.isfile(path_csv):
                shutil.copyfile(path_csv, to_foldercsv + "/" + n)
                count_copy = count_copy + 1
            else:
                print("The file is does not exist", path_csv)

    print("len(to_foldercsv) - csv", count_copy) 
    print("The files is does not exist", len(path_jsons) - count_copy)   

def copy_files_by_folder(path_from_folder, path_to_folder):
    path = "../data/json_OpenPose/OpenposeJson_x_7_c3/227969_7_c3_*.json"
    to_path = "../data/json_OpenPose/227969_7_c3"
    json_list = glob.glob(path)
    json_one_video_list = []
    for n in json_list:
        json_one_video_list.append(n)
        copyfile(n, "../data/json_OpenPose/227969_7_c3/"+os.path.split(n)[-1])
        #print(n, "../data/json_OpenPose/227969_7_c3/"+os.path.split(n)[-1])

    #print("json_one_video_list", json_one_video_list)

def set_csv_list(pathcsv_list, pathjson_list):
    videolist_csv = []
    #if len_json<=len_csv:
    for i, n in enumerate(pathcsv_list):
        d = os.path.split(pathcsv_list[i])
        d = d[1].replace(".csv", "")
        videolist_csv.append(d) 
                   
    #else:
    videolist_json = []
    for i, n in enumerate(pathjson_list):
        d = os.path.split(pathjson_list[i])
        d = d[1].replace(".json", "")
        videolist_json.append(d)

    # https://www.datacamp.com/community/tutorials/sets-in-python?utm_source=adwords_ppc&utm_medium=cpc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=278443377095&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1011984&gclid=CjwKCAiAvriMBhAuEiwA8Cs5lbjoERZE1TbGwI5YxlswN4JEZzoF9migTNuFXWh5QvBWIu-161JICxoC-oUQAvD_BwE
    videolist = list(set(videolist_csv) & set(videolist_json))
    print("set csv and json", len(videolist))
    videolist_simmetr = list(set(videolist_csv) ^ set(videolist_json))
    print("set csv ^ json", len(videolist_simmetr))
    videolist_diff_csv = list(set(videolist_csv) - set(videolist_json))
    print("set csv - json", len(videolist_diff_csv))
    videolist_diff_json = list(set(videolist_json) - set(videolist_csv))
    print("set json - csv", len(videolist_diff_json)) 
    return videolist

def copy_files_by_videolist(videolist, from_csv_folder, from_json_folder, to_csv_folder, to_json_folder):
    for n in videolist:
        path_csv = from_csv_folder + n + ".csv"
        df = pd.read_csv(path_csv)
        if round(df["fps"].iloc[0])==30:
            path_json = from_json_folder + n + ".json"
            print("df", round(df["fps"].iloc[0]))
            print(path_csv, to_csv_folder + n + ".csv")
            print(path_json,to_json_folder + n + ".json")

            copyfile(path_csv, to_csv_folder + n + ".csv")
            copyfile(path_json,to_json_folder + n + ".json")

def multiclass2binaryclass(y_full):
    uniclass = np.unique(y_full)
    y_full = np.where((y_full==2) | (y_full==4) , 1, 0)
    return y_full

if __name__ == '__main__':
    """
    #path_json = config.PATH_JSON_WLM + "319361_7_c3.mp4.json"
    path_json = config.PATH_JSON_WLM + "1000095_5_3.mp4.json"
    y = from_json_supervisely_WLM(path_json)
    print("y", y)
    """

    """
    x_full, y_full = from_json_csv_by_folders(config.PATH_CSV_SKELET, config.PATH_JSON_WLM, mode=config.MODE)
    print("y_full.shape", y_full.shape)
    print("np.unique(y_full)", np.unique(y_full))
    print("y_full", y_full)
    y_full = multiclass2binaryclass(y_full)
    print("y_full.shape", y_full.shape)
    print("np.unique(y_full)", np.unique(y_full))
    print("y_full", y_full)
    """

    """
    csvlist = glob.glob(config.PATH_CSV_SKELET_X_X_CX + '/*.csv')
    jsonlist = glob.glob(config.PATH_JSON_WLM_X_X_CX + '/*.json')
    videolist = set_csv_list(csvlist, jsonlist)
    print("csv_json_list",np.array(videolist).shape)
    from_csv_folder = config.PATH_CSV_SKELET_X_X_CX
    from_json_folder = config.PATH_JSON_WLM_X_X_CX
    to_csv_folder = "/home/cv2020/YAYAY/GitHub/slsru_ml_tag/data/csv_slsru_skelet_v0_1_0/x_x_cx_30FPS/"
    to_json_folder = "/home/cv2020/YAYAY/GitHub/slsru_ml_tag/data/json/WeaklyLabeledMovement/x_x_cx_30FPS/"
    copy_files_by_videolist(videolist, from_csv_folder, from_json_folder, to_csv_folder, to_json_folder)
    """
    
    #from_json_csv_by_folders(config.PATH_CSV_SKELET_X_X_CX, config.PATH_JSON_WLM_X_X_CX, mode="fast", path_csv="")

    """
    csvlist = glob.glob(config.PATH_CSV_SKELET_X_X_CX + '/*.csv')
    jsonlist = glob.glob(config.PATH_JSON_WLM_X_X_CX + '/*.json')
    videolist = set_csv_list(csvlist, jsonlist)
    print("csv_json_list",videolist)
    """

    """
    csvlist = glob.glob(config.PATH_CSV_SKELET_X_X_CX + '/*.csv')
    x,fps = from_csv_slsru_skelet_v0_1_0_fps_df(csvlist[0], config.SELECT_FEATURES)
    print(csvlist[0], x.shape, fps)
    """

    """
    #path_folder = "../data/json/WeaklyLabeledMovement/sxx_9and11_c5_trans/*.json"
    #path_folder = "../data/json/WeaklyLabeledMovement/x_x_cx/*.json"
    #path_folder = "../data/json/WeaklyLabeledMovement/x_x_cx_interpol_30FPS/*.json"
    path_folder = "../data/json/WeaklyLabeledMovement/sxx_9and11_c5_trans/*.json"
    wlm_jsons_validation(path_folder)
    """

    #path_folder_from_json = "../data/json/WeaklyLabeledMovement_v2_json/4and5and6/ann/"
    #path_folder_from_csv = "../data/csv_slsru_skelet_v0_1_0/burkova1006/x_5_x/"
    #path_to_csv = "../data/csv/from_json_csv_by_folders/"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+".csv"
    #X, Y = from_json_csv_by_folders(path_folder_from_csv, path_folder_from_json, path_csv=path_to_csv, mode="fast")
    
    #path_folder = "../data/json/WeaklyLabeledMovement/x_x_cx/"
    #from_foldercsv = "/media/cv2020/Data10tb/Cloud/Google/0_sl_dataset/sl_word/meta/csv_slsru_skelet_v0_1_0/burkova1006_without_folder"
    #to_foldercsv = "../data/csv_slsru_skelet_v0_1_0/x_x_cx"
    #copy_filecsvs_by(path_folder, from_foldercsv, to_foldercsv, log=True)

    """
    csvlist = glob.glob(config.PATH_CSV_SKELET_X_X_CX + '/*.csv')
    x,fps = from_csv_slsru_skelet_v0_1_0_df(csvlist[0], config.SELECT_FEATURES,fps=True)
    print(csvlist[0], x.shape, fps)
    """

    """
    path_from_folder = "../data/json_OpenPose/OpenposeJson_x_7_c3/227969_7_c3_*.json"
    path_to_folder = "../data/json_OpenPose/227969_7_c3"
    copy_files_by_folder(path_from_folder, path_to_folder)
    """
    pass
