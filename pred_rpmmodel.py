import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers, Input, Model
from keras.layers import MaxPooling2D, Flatten, Dense, BatchNormalization ,GlobalAveragePooling2D
from keras import backend as K
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from matplotlib import pyplot as plt
from Video_Generator_myline import DataGenerator_RPM_175fps

def root_mean_squared_error(y_true, y_pred):
    msle = tf.keras.losses.MeanSquaredLogarithmicError()
    return K.sqrt(msle(y_true, y_pred))

# img_input = Input(shape=(48, 48, 3), name="original_img")
# x = layers.Conv2D(48, 3, activation="relu", padding="same")(img_input)
# output = layers.Conv2D(48, 3, activation="relu", padding="same")(x)
# my_layer = Model(img_input, output, name="submodel")
#
# first_input = Input(shape=(48, 48, 3), name="first_img")
# first_output = my_layer(first_input)
# second_input = Input(shape=(48, 48, 3), name="second_img")
# second_output = my_layer(second_input)
# third_input = Input(shape=(48, 48, 3), name="third_img")
# third_output = my_layer(third_input)
# fourth_input = Input(shape=(48, 48, 3), name="fourth_img")
# fourth_output = my_layer(fourth_input)
# fifth_input = Input(shape=(48, 48, 3), name="fifth_img")
# fifth_output = my_layer(fifth_input)
#
#
# features_map = layers.Concatenate(axis=-1)(
#     [first_output, second_output, third_output, fourth_output, fifth_output]
# )
# features_map = MaxPooling2D(pool_size=(2, 2))(features_map)
#
# x = layers.Conv2D(96, 3, activation="relu", kernel_initializer='he_normal')(features_map)
# x = layers.Conv2D(96, 3, activation="relu", kernel_initializer='he_normal')(x)
# x = Flatten()(x)
# x = Dense(2048, activation="relu", kernel_initializer='he_normal')(x)
# x = Dense(1024, activation="relu", kernel_initializer='he_normal')(x)
# output = Dense(1,activation="sigmoid",name="rpm", kernel_initializer="glorot_normal")(x)
#
# model = Model(
#     inputs=[first_input, second_input, third_input, fourth_input, fifth_input],
#     outputs=output
# )
img_input = Input(shape=(48, 48, 3), name="original_img")
x = layers.Conv2D(32, 3, activation="relu", padding="same")(img_input)
output = layers.Conv2D(32, 3, activation="relu", padding="same")(x)
my_layer = Model(img_input, output, name="submodel")

first_input = Input(shape=(48, 48, 3), name="first_img")
first_output = my_layer(first_input)
second_input = Input(shape=(48, 48, 3), name="second_img")
second_output = my_layer(second_input)
third_input = Input(shape=(48, 48, 3), name="third_img")
third_output = my_layer(third_input)
fourth_input = Input(shape=(48, 48, 3), name="fourth_img")
fourth_output = my_layer(fourth_input)
fifth_input = Input(shape=(48, 48, 3), name="fifth_img")
fifth_output = my_layer(fifth_input)
sixth_input = Input(shape=(48, 48, 3), name="sixth_img")
sixth_output = my_layer(sixth_input)
seventh_input = Input(shape=(48, 48, 3), name="seventh_img")
seventh_output = my_layer(seventh_input)
eighth_input = Input(shape=(48, 48, 3), name="eighth_img")
eighth_output = my_layer(eighth_input)
nineth_input = Input(shape=(48, 48, 3), name="nineth_img")
nineth_output = my_layer(nineth_input)
ten_input = Input(shape=(48, 48, 3), name="ten_img")
ten_output = my_layer(ten_input)

features_map = layers.Concatenate(axis=-1)(
    [first_output, second_output, third_output, fourth_output, fifth_output, sixth_output, seventh_output, eighth_output, nineth_output, ten_output]
)
features_map = MaxPooling2D(pool_size=(2, 2))(features_map)

x = layers.Conv2D(32, 3, activation="relu", kernel_initializer='he_normal')(features_map)
x = layers.Conv2D(32, 3, activation="relu", kernel_initializer='he_normal')(x)
#x = layers.Conv2D(32, 3, activation="relu", kernel_initializer='he_normal')(x)
#x = Flatten()(x)
Gap = GlobalAveragePooling2D()(x)
# x = Dense(128, activation="relu", kernel_initializer='he_normal')(x)
Gap = Dense(32, activation="relu", kernel_initializer='he_normal')(Gap)
output = Dense(1,name="rpm", kernel_initializer="glorot_normal")(Gap)

model = Model(
    inputs=[first_input, second_input, third_input, fourth_input, fifth_input, sixth_input, seventh_input, eighth_input, nineth_input, ten_input],
    outputs=output
)
# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
#               loss=root_mean_squared_error,
#               metrics=[tf.keras.metrics.RootMeanSquaredError()]
#               )
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=root_mean_squared_error,
              #metrics=['accuracy']
              )

model.summary()

# tf.keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
# tf.keras.utils.plot_model(my_layer, "layers.png", show_shapes=True)

# gettargetcsv_RPM

csv_df = pd.read_csv(r"D:\My_Files\zly_python_file\baseball\python\175fps\5fold\175fps_fivefold_trainset0.csv")
save_model_path = r'D:\My_Files\zly_python_file\baseball\python\175fps\RPM_h5\10_175fps_2022_09_07_train0_rmse.h5'
save_plot_loss_path = r'D:\My_Files\zly_python_file\baseball\python\175fps\plot\10_175fps_2022_09_07_train0_rmse.png'

csv_df = shuffle(csv_df)
train_data, val_data = train_test_split(csv_df, train_size=0.8, random_state=175)
traingen = DataGenerator_RPM_175fps(dataFrame = train_data,batch_size=64,Size=48)
valgen = DataGenerator_RPM_175fps(dataFrame = val_data,batch_size=64,Size=48)
# traingen = DataGenerator_myRPM(dataFrame = train_data,batch_size=64,Size=48)
# valgen = DataGenerator_myRPM(dataFrame = val_data,batch_size=256,Size=48)

checkpoint_path = r"D:\My_Files\zly_python_file\baseball\python\175fps\RPM_ckpt\10_175fps_2022_09_07_train0_rmse.h5"
callback_checkpoint = [
    #tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, mode="min", restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, monitor="val_loss", save_best_only=True, mode="min",period=5),
    # tf.keras.callbacks.TensorBoard(log_dir='./logs'),
]


history = model.fit(
    traingen,
    verbose=1,
    epochs=150, # Change this to a larger number to train for longer
    validation_data= valgen,
    callbacks=[callback_checkpoint]
)
model.save(save_model_path)
plt.rcParams["figure.figsize"] = (25, 10)
plt.plot(history.history['loss'][10:])
plt.plot(history.history['val_loss'][10:])
plt.legend(['loss', 'val_loss'])
plt.savefig(save_plot_loss_path)
plt.show()