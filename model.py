import efficientnet.tfkeras as efn
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import regularizers

# weight decay coefficient
alpha = 1e-5
# kernel initializer
k_init = "he_normal"

# Loss Functions
def dice_loss(y_true, y_pred):
    smooth = 1.0
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2.0 * K.sum(intersection) + smooth) / (
        K.sum(y_true_f) + K.sum(y_pred_f) + smooth
    )
    return 1.0 - score


def ce_dice_loss(y_true, y_pred):
    return tf.losses.BinaryCrossentropy()(y_true, y_pred) + dice_loss(y_true, y_pred)


def ConvBnAct(
    filters,
    kernel_size,
    strides=(1, 1),
    padding="same",
    dilation_rate=(1, 1),
    kernel_initializer=k_init,
    kernel_regularizer=regularizers.l2(alpha),
    activation=None,
    use_bn=False,
    name=None,
):
    conv_name, bn_name, act_name = None, None, None
    if name:
        conv_name = name + "_Conv2D"
        if use_bn:
            bn_name = name + "_BN"
        if activation:
            act_name = name + activation

    def ret(x):
        x = Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding=padding,
            dilation_rate=dilation_rate,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            use_bias=not (use_bn),
            name=conv_name,
        )(x)

        if use_bn:
            x = BatchNormalization(name=bn_name)(x)

        if activation:
            x = Activation(activation, name=act_name)(x)

        return x

    return ret


def deep_supervision_block(
    inp, out_classes, out_shape, name="", last_layer_name="", upscale_factor=2
):
    inp = Conv2D(
        out_classes,
        (1, 1),
        strides=(1, 1),
        kernel_initializer=k_init,
        name=name + "Conv1",
        kernel_regularizer=regularizers.l2(alpha),
    )(inp)
    inp = Activation("sigmoid", name=name + "Act1")(inp)
    if upscale_factor != 1:
        inp = UpSampling2D(
            size=(upscale_factor, upscale_factor),
            name=name + "Upsamplingdsv",
            interpolation="bilinear",
        )(inp)
    inp = Lambda(lambda x: x, name=last_layer_name)(inp)
    return inp


def upsample_skip(skip=None, upsampling_type="nearest", name=None):
    def ret(x, skip=skip):
        x = UpSampling2D((2, 2), interpolation=upsampling_type, name=f"{name}_up")(x)
        if skip is not None:
            x = Concatenate(name=f"{name}_conc")([x, skip])
        return x

    return ret


def conv_stack(filters, name=None):
    def ret(x):
        x = ConvBnAct(
            filters,
            (3, 3),
            strides=(1, 1),
            kernel_initializer=k_init,
            activation="relu",
            use_bn=True,
            name=f"{name}_ConvStack1",
        )(x)
        x = ConvBnAct(
            filters,
            (3, 3),
            strides=(1, 1),
            kernel_initializer=k_init,
            activation="relu",
            use_bn=True,
            name=f"{name}_ConvStack2",
        )(x)

        return x

    return ret


def residual_block(filters, name=None):
    def ret(x):
        _, _, _, in_c = K.int_shape(x)
        if in_c != filters:
            x = ConvBnAct(
                filters, (1, 1), activation=None, use_bn=False, name=f"{name}_Res1x1",
            )

        skip = x

        x = ConvBnAct(
            filters, (3, 3), activation="relu", name=f"{name}_ResConvStack1",
        )(x)
        x = ConvBnAct(
            filters, (3, 3), activation="relu", name=f"{name}_ResConvStack2",
        )(x)
        x = Add(name=f"{name}_ResAdd")([x, skip])

        return x

    return ret


def residual_conv_stack(filters, name=None):
    def ret(x):
        x = residual_block(filters, name=f"{name}_A")(x)
        x = residual_block(filters, name=f"{name}_B")(x)
        return x

    return ret


# Model Function
def get_model(input_height, input_width, n_ch):
    """Generates the crack segmentation model as specified in the paper: 'Optimized Deep Encoder-Decoder Methods for Crack Segmentation'.
       This model uses an EfficientNet B5 backbone. 

    Args:
        input_height (int): input height of the model
        input_width (int): input width of the model
        n_ch (int): number of channels of the input

    Returns:
        tensorflow.keras.Model: Crack segmentation model (already compiled)
    """

    filters = (256, 128, 64, 32, 16)

    backbone = efn.EfficientNetB5(
        weights="imagenet",
        input_shape=(input_height, input_width, 3),
        include_top=False,
    )
    # some adaptions from https://github.com/qubvel/segmentation_models/blob/master/segmentation_models/models/unet.py
    x = backbone.output
    skip_connection_layers = (
        "block6a_expand_activation",
        "block4a_expand_activation",
        "block3a_expand_activation",
        "block2a_expand_activation",
    )

    # extract skip connections
    skips = [backbone.get_layer(name=i).output for i in skip_connection_layers]

    # add a conv after the input to extract some information from the original scale
    inp = backbone.layers[0].output
    inp = conv_stack(16, name="conv_inp")(inp)

    dsvs = []

    for i in range(len(filters)):
        name = f"decoderBlock{i}"
        if i < len(skips):
            skip = skips[i]
        else:
            skip = inp
        x = upsample_skip(filters[i], name=name)(x, skip)

        if i < len(skips):
            x = ConvBnAct(
                filters[i],
                (3, 3),
                activation="relu",
                use_bn=True,
                name=f"{name}preconv",
            )(x)
            x = residual_conv_stack(filters[i], name=name)(x)

            dsvs.append(x)
        else:
            x = conv_stack(filters[i], name="inp1" + name)(x)

    x = Conv2D(
        filters=1,
        kernel_size=(3, 3),
        padding="same",
        use_bias=True,
        kernel_initializer=k_init,
        kernel_regularizer=regularizers.l2(alpha),
        name="output_conv",
    )(x)
    x = Activation("sigmoid", name="main_out")(x)

    interim_model = Model(inputs=backbone.input, outputs=[x,] + dsvs)

    # add weight decay
    for layer in interim_model.layers:
        for attr in ["kernel_regularizer"]:
            if hasattr(layer, attr):
                setattr(layer, attr, keras.regularizers.l2(alpha))

    inputs = Input(shape=(input_height, input_width, n_ch))

    if n_ch == 1:
        inp = Lambda(lambda x: K.tile(x, (1, 1, 1, 3)))(inputs)
    else:
        inp = inputs

    interim_outputs = interim_model(inp)

    outputs = []

    upscale_factors = [16, 8, 4, 2]
    for ix, o in enumerate(interim_outputs):
        if ix == 0:
            outputs.append(Lambda(lambda x: x, name="main_out")(o))
        else:
            o = deep_supervision_block(
                o,
                1,
                name="dsv" + str(ix + 1),
                out_shape=(input_height, input_width),
                last_layer_name="aux" + str(ix + 1),
                upscale_factor=upscale_factors[ix - 1],
            )
            outputs.append(o)
    sgd = SGD(lr=0.001, momentum=0.9, nesterov=False)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=sgd, loss=ce_dice_loss, metrics=["accuracy"])

    return model


if __name__ == "__main__":
    model = get_model(288, 288, 3)
    model.summary()

