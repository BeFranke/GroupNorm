from typing import Union

import tensorflow as tf


class GroupNormalization(tf.keras.layers.Layer):
    def __init__(self,
                 groups: int = 16,
                 eps: float = 1e-5,
                 gamma_initializer: Union[None, str, tf.keras.initializers.Initializer] = 'ones',
                 beta_initializer: Union[None, str, tf.keras.initializers.Initializer] = 'zeros',
                 gamma_regularizer: Union[None, tf.keras.regularizers.Regularizer] = tf.keras.regularizers.L2(0.0001),
                 beta_regularizer: Union[None, tf.keras.regularizers.Regularizer] = tf.keras.regularizers.L2(0.0001),
                 gamma_constraint: Union[None, tf.keras.constraints.Constraint] = None,
                 beta_constraint: Union[None, tf.keras.constraints.Constraint] = None):
        super().__init__()
        self.eps = eps
        self.gamma = None
        self.beta = None
        self.beta_constraint = beta_constraint
        self.gamma_constraint = gamma_constraint
        self.beta_regularizer = beta_regularizer
        self.gamma_regularizer = gamma_regularizer
        self.beta_initializer = beta_initializer
        self.gamma_initializer = gamma_initializer
        self.groups = groups
        self._moment_axes = None

    def build(self, input_shape):
        if tf.keras.backend.image_data_format() == "channels_first":
            N, C, H, W = input_shape

            assert C % self.groups == 0, f"Groups need to evenly divide the input channels, " \
                                         f"but got {C} input channels and {self.groups} groups!"

            self.gamma = self.add_weight(
                name="gamma",
                shape=(1, C, 1, 1),
                trainable=True,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(1, C, 1, 1),
                trainable=True,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint
            )
            self.group = tf.keras.layers.Reshape((self.groups, C // self.groups, H, W))
            self.degroup = tf.keras.layers.Reshape((C, H, W))
            self._moment_axes = [2, 3, 4]
        else:
            N, H, W, C = input_shape

            assert C % self.groups == 0, f"Groups need to evenly divide the input channels, " \
                                         f"but got {C} input channels and {self.groups} groups!"

            self.gamma = self.add_weight(
                name="gamma",
                shape=(1, 1, 1, C),
                trainable=True,
                initializer=self.gamma_initializer,
                regularizer=self.gamma_regularizer,
                constraint=self.gamma_constraint
            )
            self.beta = self.add_weight(
                name="beta",
                shape=(1, 1, 1, C),
                trainable=True,
                initializer=self.beta_initializer,
                regularizer=self.beta_regularizer,
                constraint=self.beta_constraint
            )

            self.group = tf.keras.layers.Reshape((H, W, self.groups, C // self.groups))
            self.degroup = tf.keras.layers.Reshape((H, W, C))
            self._moment_axes = [1, 2, 4]

        self.built = True

    @tf.function
    def call(self, x: tf.Tensor, **kwargs):
        x = self.group(x)

        mean, var = tf.nn.moments(x, self._moment_axes, keepdims=True)
        x = (x - mean) / tf.sqrt(var + self.eps)

        x = self.degroup(x)

        return x * self.gamma + self.beta
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "groups": self.groups,
            "eps": self.eps,
            "gamma_initializer": self.gamma_initializer,
            "beta_initializer": self.beta_initializer,
            "gamma_regularizer": self.gamma_regularizer,
            "beta_regularizer": self.beta_regularizer,
            "gamma_constraint": self.gamma_constraint,
            "beta_constraint": self.beta_constraint
        })
        return config


def test_numeric():
    assert tf.keras.backend.image_data_format() == "channels_last", \
        "This test was built for the 'channels_last' image data format!"
    X = tf.concat(
        [
            tf.random.normal((1, 32, 32, 2), mean=10, stddev=10),
            tf.random.normal((1, 32, 32, 2), mean=5, stddev=50)
        ], axis=1
    )
    gn = GroupNormalization(groups=2)
    gn.build((None, 32, 32, 4))
    res = gn(X)
    g1, g2 = tf.split(res, axis=3, num_or_size_splits=2)
    mean1, var1 = tf.nn.moments(g1, axes=[0, 1, 2, 3])
    mean2, var2 = tf.nn.moments(g2, axes=[0, 1, 2, 3])
    tf.debugging.assert_near(mean1, 0)
    tf.debugging.assert_near(var1, 1)
    tf.debugging.assert_near(mean2, 0)
    tf.debugging.assert_near(var2, 1)


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
