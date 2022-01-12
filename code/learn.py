# cd Logs\train
# tensorboard --logdir=Logs\train

# One-to-Many, Many-to-One and Many-to-Many LSTM Examples in Keras
# https://wandb.ai/ayush-thakur/dl-question-bank/reports/One-to-Many-Many-to-One-and-Many-to-Many-LSTM-Examples-in-Keras--VmlldzoyMDIzOTM
# Human Activity Recognition using Deep Learning Models on Smartphones and Smartwatches Sensor Data
# https://arxiv.org/pdf/2103.03836.pdf
# Ungraded Lab: Feature Engineering with Accelerometer Data
# https://github.com/YAYAYru/coursera-machine-learning-engineering-for-prod-mlops-specialization/blob/main/C2%20-%20Machine%20Learning%20Data%20Lifecycle%20in%20Production/Week%204/C2_W4_Lab_2_Signals.ipynb
# C4/W3/ungraded_labs/C4_W3_Lab_2_LSTM.ipynb
# https://github.com/https-deeplearning-ai/tensorflow-1-public/blob/main/C4/W3/ungraded_labs/C4_W3_Lab_2_LSTM.ipynb
# Нужно вопрос на форуме
# https://ai.stackexchange.com/questions/6863/how-to-adapt-rnns-to-variable-frequency-framerate-of-inputs



from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, GRU, TimeDistributed, RepeatVector
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import TensorBoard

import datetime



import config
#test
#step_x = step_x.reshape(step_x.shape[0], step_x.shape[2]*step_x.shape[1],step_x.shape[3])

def split_train_val_test(step_x, step_y, test_size):
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(step_x, step_y, test_size=test_size*2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=42)
    return X_train, X_val, X_test, Y_train, Y_val, Y_test
    

def split_to_categorical(step_x, step_y):
    print("def split_model_fit(step_x, step_y, path_model)")
    X_train, X_val_test, Y_train, Y_val_test = train_test_split(step_x, step_y, test_size=config.TEST_SIZE*2, random_state=42)
    X_val, X_test, Y_val, Y_test = train_test_split(X_val_test, Y_val_test, test_size=0.5, random_state=42)

    Y_train = to_categorical(Y_train)
    Y_val = to_categorical(Y_val)
    #Y_test = to_categorical(Y_test)
    #inp_s = step_x.shape[1]
    #inp_s2 = step_x.shape[2]

    print("X_train.shape", X_train.shape)
    print("X_val.shape", X_val.shape)
    print("X_test.shape", X_test.shape)
    print("Y_train.shape", Y_train.shape)
    print("Y_val.shape", Y_val.shape)
    print("Y_test.shape", Y_test.shape)

    return X_train, X_val, X_test, Y_train, Y_val, Y_test
def model_fn(x_shape, y_shape):
    print("def model_fn(x_shape, y_shape)")
    print("x_shape", x_shape)
    print("y_shape", y_shape)

    if config.MODEL_NET=="model_fn_v1":
        model = model_fn_v1(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v1_1":
        model = model_fn_v1_1(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v1_2":
        model = model_fn_v1_2(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v1_3":
        model = model_fn_v1_3(x_shape, y_shape)

    if config.MODEL_NET=="model_fn_v2":
        model = model_fn_v2(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v3":
        model = model_fn_v3(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v2_GRU":
        model = model_fn_v2_GRU(x_shape, y_shape)
    if config.MODEL_NET=="model_fn_v1_timedistributed":
        model = model_fn_v1_timedistributed(x_shape, y_shape)
    if config.MODEL_NET=="model_many_to_one_2class_v1":
        model = model_many_to_one_2class_v1(x_shape, y_shape)
    if config.MODEL_NET=="model_many_to_many_2class_v1":
        model = model_many_to_many_2class_v1(x_shape, y_shape)
    if config.MODEL_NET=="model_many_to_one_2class_v2":
        model = model_many_to_one_2class_v2(x_shape, y_shape)
    if config.MODEL_NET=="model_many_to_one_2class_v3":
        model = model_many_to_one_2class_v3(x_shape, y_shape)

    return model

# подходит c преобразованием по углам
def model_fn_v1(x_shape, y_shape):

    model = Sequential()
    model.add(LSTM(512, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_fn_v1_1(x_shape, y_shape):

    model = Sequential()
    model.add(GRU(512, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(GRU(128, return_sequences=True, activation='relu'))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_fn_v1_2(x_shape, y_shape):
    model = Sequential()
    #model.add(LSTM(512, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))    
    model.add(Bidirectional(LSTM(512, return_sequences=True, activation='relu'), input_shape=(x_shape[1],x_shape[2])))    
    #model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Bidirectional(LSTM(128, return_sequences=True, activation='relu')))
    model.add(Bidirectional(LSTM(64, return_sequences=False, activation='relu')))
    #model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_fn_v1_3(x_shape, y_shape):
    model = Sequential()
    #model.add(LSTM(512, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))    
    model.add(Bidirectional(GRU(512, return_sequences=True, activation='relu'), input_shape=(x_shape[1],x_shape[2])))    
    #model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(Bidirectional(GRU(128, return_sequences=True, activation='relu')))
    model.add(Bidirectional(GRU(64, return_sequences=False, activation='relu')))
    #model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# подходит без преобразования по углам
def model_fn_v2(x_shape, y_shape):

    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
# If windows_size=40 then here
def model_fn_v3(x_shape, y_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_fn_v2_GRU(x_shape, y_shape):
    model = Sequential()
    model.add(GRU(32, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(GRU(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
    
def model_fn_v1_timedistributed(x_shape, y_shape):
    model = Sequential()
    model.add(LSTM(512, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1], activation='softmax'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_manytomany_v1(x_shape, y_shape):
    model = Sequential()
    model.add(LSTM(100, input_shape=(x_shape[1], x_shape[2]),return_sequences=True))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(LSTM(y_shape[2],return_sequences=True))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def model_many_to_many_encoder_decoder_v1(x_shape, y_shape):
    model = Sequential()
    # encoder layer
    model.add(LSTM(100, activation='relu', input_shape=(x_shape[1], x_shape[2])))
    # repeat vector
    model.add(RepeatVector(x_shape[1]))
    # decoder layer
    #model.add(Bidirectional(LSTM(100, activation='relu', return_sequences=True)))
    model.add(LSTM(100, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(y_shape[2], activation="softmax")))
    model.compile(optimizer='adam', loss='mse')
    #model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())    
    return model

def model_many_to_one_2class_v1(x_shape, y_shape):
    print("model_many_to_one_2class_v1(x_shape, y_shape)")
    print("x_shape, y_shape", x_shape, y_shape)
    model = Sequential()
    model.add(LSTM(32, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(y_shape[1],activation='sigmoid'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_many_to_one_2class_v2(x_shape, y_shape):
    print("model_many_to_one_2class_v2(x_shape, y_shape)")
    print("x_shape, y_shape", x_shape, y_shape)
    model = Sequential()
    model.add(LSTM(31, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(31, return_sequences=False, activation='relu'))
    model.add(Dense(31, activation='relu'))
    model.add(Dense(y_shape[1],activation='sigmoid'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_many_to_one_2class_v3(x_shape, y_shape):
    print("model_many_to_one_2class_v3(x_shape, y_shape)")
    print("x_shape, y_shape", x_shape, y_shape)
    model = Sequential()
    model.add(LSTM(61, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(31, return_sequences=False, activation='relu'))
    model.add(Dense(31, activation='relu'))
    model.add(Dense(y_shape[1],activation='sigmoid'))
    model.summary()
    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def model_many_to_many_2class_v1(x_shape, y_shape):
    print("model_many_to_one_2class_v1(x_shape, y_shape)")
    print("x_shape, y_shape", x_shape, y_shape)
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(x_shape[1],x_shape[2])))
    model.add(LSTM(64, return_sequences=True, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(TimeDistributed(Dense(y_shape[2])))
    model.summary()

    model.compile(optimizer=Adam(learning_rate=config.LEARNING_RATE), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def fit(model, X_train, X_val, Y_train, Y_val, path_model, callbacks=0):


    history = model.fit(X_train,
                    Y_train,
                    epochs=config.EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    validation_data = (X_val, Y_val),
                    callbacks=callbacks
                    )
    res = model.predict(X_val)
    model.save(path_model)
    print("path_model", path_model)


def fit_temp(model, X_train, X_val, Y_train, Y_val, path_model):
    #log_dir = os.path.join('Logs')
    #tb_callback = TensorBoard(log_dir=log_dir)
    log_dir = "../logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(log_dir=log_dir)

    #callbacks_list = [checkpoint, learning_rate_reduction]
    history = model.fit(X_train,
                    Y_train,
                    epochs=config.EPOCHS,
                    batch_size=config.BATCH_SIZE,
                    validation_data = (X_val, Y_val),
                    #callbacks=callbacks_list
                    #callbacks=tensorboard_callback
                    )
    res = model.predict(X_val)
    model.save(path_model)
    print("path_model", path_model)
    
    
    plt.plot(history.history['accuracy'], 
         label='Доля верных ответов на обучающем наборе')
    plt.plot(history.history['val_accuracy'], 
         label='Доля верных ответов на проверочном наборе')
    plt.xlabel('Эпоха обучения')
    plt.ylabel('Доля верных ответов')
    plt.legend()
    plt.savefig(config.PATH_IMAGE_HISTORY)
    plt.cla()
