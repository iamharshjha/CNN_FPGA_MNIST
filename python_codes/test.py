from tensorflow import keras

model = keras.models.load_model("./trained_model.h5" , compile = True)
model.summary()
