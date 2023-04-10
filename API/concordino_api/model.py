from tensorflow import keras

def load_model(path):
    return keras.models.load_model(path)

def load_prediciton_model(model):
    return keras.models.Model(
        model.get_layer(name='image').input,
        model.get_layer(name='dense2').output
    )
