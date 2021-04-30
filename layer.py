import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self, G: int = 32, eps: float = 1e-5):
        super().__init__()
        self.G = G
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.moment_axes = None

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == "channels_first":
            N, C, H, W = input_shape
            self.gamma = self.add_weight(
                name="gamma",
                shape=(1, C, 1, 1),
                trainable=True
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(1, C, 1, 1),
                trainable=True
            )
            self.group = tf.keras.layers.Reshape((self.G, C // self.G, H, W))
            self.degroup = tf.keras.layers.Reshape((C, H, W))
            self.moment_axes = [2, 3, 4]
        else:
            N, H, W, C = input_shape
            self.gamma = self.add_weight(
                name="gamma",
                shape=(1, 1, 1, C),
                trainable=True
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(1, 1, 1, C),
                trainable=True
            )

            self.group = tf.keras.layers.Reshape((H, W, self.G, C // self.G))
            self.degroup = tf.keras.layers.Reshape((H, W, C))
            self.moment_axes = [1, 2, 4]

        self.built = True

    @tf.function
    def call(self, x: tf.Tensor, **kwargs):
        x = self.group(x)

        mean, var = tf.nn.moments(x, self.moment_axes, keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)

        x = self.degroup(x)

        return x * self.gamma + self.beta


def test_numeric():
    X = tf.concat(
        [
            tf.random.normal((1, 2, 32, 32), mean=10, stddev=10),
            tf.random.normal((1, 2, 32, 32), mean=50, stddev=0.1)
        ], axis=1
    )
    gn = GroupNormalization(G=2)
    gn.build((None, 4, 32, 32))
    gn.gamma = tf.ones(shape=(1, 4, 1, 1))
    gn.beta = tf.zeros(shape=(1, 4, 1, 1))
    res = gn(X)
    g1, g2 = tf.split(res, axis=1, num_or_size_splits=2)
    mean1, var1 = tf.nn.moments(g1, axes=[0, 1, 2, 3])
    mean2, var2 = tf.nn.moments(g2, axes=[0, 1, 2, 3])
    tf.debugging.assert_near(mean1, 0, atol=0.01, rtol=0.01)
    tf.debugging.assert_near(var1, 1, atol=0.01, rtol=0.01)
    tf.debugging.assert_near(mean2, 0, atol=0.01, rtol=0.01)
    tf.debugging.assert_near(var2, 1, atol=0.01, rtol=0.01)


def test_interface():
    # network will ot learn anything, as the data is random.
    # But it tests if GroupNormalization works with other keras layers
    data = tf.random.normal((1, 32, 32, 8))
    y = tf.ones((1, 1))
    inp = tf.keras.Input((32, 32, 8))
    x = tf.keras.layers.Conv2D(64, 3)(inp)
    x = GroupNormalization(8)(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inp, outputs=x)
    model.compile(optimizer='SGD', loss='MSE')
    model.fit(data, y, epochs=1)


if __name__ == "__main__":
    print("testing computations...")
    test_numeric()
    print("passed!")

    print("testing tf-interface...")
    test_interface()
    print("passed!")
