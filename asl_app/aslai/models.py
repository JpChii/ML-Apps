# Modelling

import tensorflow as tf
from tensorflow.keras import applications

def create_base_model(include_top: bool = False, trainable: bool = True, layers_to_tune: int = 30) -> tf.keras.Model:
    """
    Function to load ResNet101 model from tensorflow.keras.applications as a model for transfer learning

    Args:
        include_top (bool): Variable to include or not include the top layers in the model
        trainable (bool): Variable to define whether the model is trainable or not
        layers_to_tune (int): Number of layers to be unfrozen when trainable is True

    Returns:
        tf.keras.Model
    """

    base_model = applications.ResNet101(include_top=include_top)

    # Checking whether the base model is trainable and number of layers to unfroze are available
    if trainable == True and layers_to_tune != 0:
        base_model.trainable = trainable
        # Except the layer_to_tune freezing all other layers
        for layer in base_model.layers[:-layers_to_tune]:
            layer.trainable = False

    return base_model

print(create_base_model())