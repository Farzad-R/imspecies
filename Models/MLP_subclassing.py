from typing import List
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout


# Model subclassing
class MlpClassifier(tf.keras.Model):
    def __init__(
        self,
        n_features,
        n_classes,
        hidden_sizes: List[int],
        activation="relu",
        dropout_rate: float = 0.0,
    ):
        super(MlpClassifier, self).__init__()
        self.input_fc = Dense(hidden_sizes[0], activation=activation)
        self.input_dropout = Dropout(dropout_rate)
        self.n_features = n_features

        self.n_fc = len(hidden_sizes) - 1
        for i, hidden_size in enumerate(hidden_sizes[1:]):
            fc = Dense(hidden_size, activation=activation)
            setattr(self, f"dense{i}", fc)
            setattr(self, f"dropout{i}", Dropout(dropout_rate))

        self.output_fc = Dense(n_classes, activation="softmax")

    def call(self, inputs, training=False):
        x = self.input_fc(inputs)
        if training:
            x = self.input_dropout(x)
        for i in range(self.n_fc):
            fc = getattr(self, f"dense{i}")
            x = fc(x)
            if training:
                dropout = getattr(self, f"dropout{i}")
                x = dropout(x)
        return self.output_fc(x)

    def summary(self, training=True):
        x = Input(shape=(self.n_features,))
        Model(inputs=[x], outputs=self.call(x, training=training)).summary()


class MlpBinaryClassifier(tf.keras.Model):
    def __init__(
        self,
        n_features,
        hidden_sizes: List[int],
        activation="relu",
        dropout_rate: float = 0.0,
    ):
        super(MlpBinaryClassifier, self).__init__()
        self.input_fc = Dense(hidden_sizes[0], activation=activation)
        self.input_dropout = Dropout(dropout_rate)
        self.n_features = n_features

        self.n_fc = len(hidden_sizes) - 1
        for i, hidden_size in enumerate(hidden_sizes[1:]):
            fc = Dense(hidden_size, activation=activation)
            setattr(self, f"dense{i}", fc)
            setattr(self, f"dropout{i}", Dropout(dropout_rate))

        self.output_fc = Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x = self.input_fc(inputs)
        if training:
            x = self.input_dropout(x)
        for i in range(self.n_fc):
            fc = getattr(self, f"dense{i}")
            x = fc(x)
            if training:
                dropout = getattr(self, f"dropout{i}")
                x = dropout(x)
        return self.output_fc(x)

    def summary(self, training=True):
        x = Input(shape=(self.n_features,))
        Model(inputs=[x], outputs=self.call(x, training=training)).summary()


# model = MlpClassifier()
# predictions = model(input)
# model.fit(x, y, epochs=10, batch_size=32)

