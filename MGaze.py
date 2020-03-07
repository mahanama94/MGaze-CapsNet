from tensorflow.keras.datasets import mnist
from tensorflow.keras import layers, activations, models, utils, callbacks
from capsnet.capsnet import *

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

x_train, x_test = x_train[..., np.newaxis], x_test[..., np.newaxis]
NUM_CLASSES = 10
y_train, y_test = utils.to_categorical(y_train, NUM_CLASSES), utils.to_categorical(y_test, NUM_CLASSES)

_input_shape = x_train.shape[1:]
_output_shape = y_train.shape[1:]

# CAPSULE NETWORK ARCHITECTURE ---
conv_1_spec = {
    'filters': 256,
    'kernel_size': (9, 9),
    'strides': (1, 1),
    'activation': activations.relu
}

# capsule 1 spec
num_filters = 32
dim_cap_1 = 8

conv_2_spec = {
    'filters': num_filters,
    'kernel_size': (9, 9),
    'strides': (2, 2),
    'activation': activations.relu
}

digit_caps_spec = {
    'num_caps': 10,
    'dim_caps': 16,
    'routing_iter': 3
}

# input
l1 = layers.Input(shape=_input_shape)

# initial convolution
l2 = layers.Conv2D(**conv_1_spec)(l1)

# primary caps (convolution + reshape + squash)
l3 = layers.Conv2D(**conv_2_spec)(l2)
l4 = layers.Reshape((np.prod(l3.shape[1:]) // dim_cap_1, dim_cap_1))(l3)
l5 = layers.Lambda(squash)(l4)

# digit caps (routing based on agreement -> weighted prediction)
l6 = DigitCaps(**digit_caps_spec)(l5)

# predictions (None, dim_caps)
l7 = layers.Lambda(safe_l2_norm, name='margin')(l6)

# masking layer
l8 = layers.Lambda(mask)(l6)

# decoder
d0 = layers.Flatten()(l8)
d1 = layers.Dense(512, activation='relu')(d0)
d2 = layers.Dense(1024, activation='relu')(d1)
d3 = layers.Dense(np.prod(_input_shape), activation='sigmoid')(d2)
d4 = layers.Reshape(_input_shape, name='reconstruction')(d3)

# define the model
model = models.Model(inputs=l1, outputs=[l7, d4], name='capsule_network')
model.compile(optimizer='adam', loss=[margin_loss, reconstruction_loss], loss_weights=[1e0, 5e-3],
              metrics={'margin': accuracy})

# checkpoint function to save best weights
checkpoint = callbacks.ModelCheckpoint("best_weights.hdf5", save_best_only=True)

if os.path.exists('best_weights.hdf5'):
    # load existing weights
    model.load_weights('best_weights.hdf5')
else:
    # training
    model.fit(x_train, [y_train, x_train], batch_size=50, epochs=5, validation_split=0.1, callbacks=[checkpoint])
    # load best weights
    model.load_weights('best_weights.hdf5')
    # evaluation
    model.evaluate(x_test, [y_test, x_test])


def print_results():
    indices = np.random.randint(0, len(x_test), 10)
    _n, _x, _y = len(indices), x_test[indices], y_test[indices]
    [_y_p, _x_p] = model.predict(_x)
    fig, axs = plt.subplots(ncols=5, nrows=4)
    for z in range(_n):
        i = (z // 5) * 2
        j = z % 5
        axs[i, j].imshow(np.squeeze(_x_p[z]), cmap='gray', vmin=0.0, vmax=1.0)
        axs[i, j].axis('off')
        axs[i + 1, j].imshow(np.squeeze(_x[z]), cmap='gray', vmin=0.0, vmax=1.0)
        axs[i + 1, j].axis('off')
    fig.show()


print_results()
