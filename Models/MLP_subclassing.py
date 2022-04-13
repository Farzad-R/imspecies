# import seaborn as sns
# sns.set()
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# # %matplotlib qt
from typing import List
import tensorflow as tf
from tensorflow.keras import Model
# import tensorflow.keras as keras
# from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
# from tensorflow.keras import optimizers
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import classification_report
# from sklearn.metrics import confusion_matrix
# from sklearn.utils import compute_class_weight
# from sklearn import metrics
# from sklearn.metrics import classification_report, confusion_matrix
import logging

logger = logging.getLogger(__name__)
# from __future__ import print_function

# # plot confusuin matrix
# def plot_cm(labels, predictions, p=0.5):
#     cm = confusion_matrix(labels, predictions)
#     plt.figure(figsize=(5,5))
#     sns.heatmap(cm, annot=True, fmt="d")
#     plt.title('Confusion matrix @{:.2f}'.format(p))
#     plt.ylabel('Actual label')
#     plt.xlabel('Predicted label')

# def class_weight_func(y_train=data['y_train']):
#     trainset_len = len(y_train)
#     class_weight = {}
#     y_train_not_enc = np.argmax(y_train, axis=1)
#     n_classes = len(y_train[0])
#     for i in range(len(y_train[0])):
#         class_weight[i] = 1 / (y_train_not_enc == i).sum() * trainset_len / n_classes
#     return class_weight
# ###!!!!!!!!!!!!!!!!!!!!!!!
# def early_stopping():
#     early_stopping = EarlyStopping(
#     monitor='val_recall', 
#     verbose=1,
#     patience=50,
#     mode='max',
#     restore_best_weights=True)
#     my_callbacks = [early_stopping]
#     return my_callbacks


# class MlpClassifier(keras.Model):
#     # MLP classifier
#     def build_mlp(self, **ca.kwargs_build_mlp):
#         ca = config.kwargs_architectures
#         classifier = Sequential()
#         classifier.add(Dense(units=hidden_shape[0], activation=activation , input_shape=input_size))
#         classifier.add(Dropout(dropRate, input_shape=(hidden_shape[0],)))
#         for layer in hidden_shape[1:]:
#             classifier.add(Dense(layer, activation=activation))
#             classifier.add(Dropout(dropRate, input_shape=(layer,)))
#         # Last layer
#         classifier.add(Dense(units=num_classes, activation='softmax'))
        
#         print(classifier.summary())
#         return classifier

#     def compile_mlp(self, **ca.kwargs_train_classifier):
#         ca = config.kwargs_architectures
#         classifier = build_mlp(**ca.kwargs_build_mlp)
#         if opt == 'Adam':
#             opt = optimizers.Adam(lr)
#         elif opt == 'SGD':
#             opt = optimizers.SGD(momentum=0.9, nesterov=True)
#         else:
#             print('The given optimizer is not implemented') 
        
#         if loss == 'categorical':
#             loss = tf.keras.losses.CategoricalCrossentropy(
#                 name='categorical_crossentropy')
#         elif: loss == 'binary':
#         loss = tf.keras.losses.BinaryCrossentropy(
#             name='binary_crossentropy'
#         )
#         else:
#             print('The given loss function is not implemented')
            
#         # Compiling the ANN
#         classifier.compile(optimizer=opt , loss=loss, metrics=METRICS)
#         return classifier

#     def run_mlp(self, config, class_weight=True, data, batch_size, num_epoch):
#         ca = config.kwargs_architectures
#         classifier = compile_classifier()
#         my_callbacks = early_stopping()
#         if class_wieght:
#             class_weight = class_weight_func(y_train)
#             history = classifier.fit(data['X_train'], data['y_train'], batch_size=batch_size, validation_data=(data['X_valid'], data['y_valid']), epochs=num_epoch , verbose=1, callbacks=my_callbacks, class_weight=class_weight)
#         else:
#             history = classifier.fit(data['X_train'], data['y_train'], batch_size=batch_size, validation_data=(data['X_valid'], data['y_valid']), epochs=num_epoch, verbose=1, callbacks=my_callbacks)

#             train_preds = classifier.predict(data['X_train'], batch_size=batch_size)
#             val_preds = classifier.predict(data['X_test'], batch_size=batch_size)
#             test_preds = classifier.predict(data['X_valid'], batch_size=batch_size)

#             # plotting the confusion matrices
#             plot_cm(np.argmax(y_train_enc, axis=1), np.argmax(train_preds, axis=1))
#             # pyplot.savefig('train.png', dpi=400)
#             plot_cm(np.argmax(y_valid_enc, axis=1), np.argmax(val_preds, axis=1))
#             # pyplot.savefig('valid.png', dpi=400)
#             plot_cm(np.argmax(y_test_enc, axis=1), np.argmax(test_preds, axis=1))
#             # pyplot.savefig('test.png', dpi=400)


# # Model subclassing
class MlpClassifier(tf.keras.Model):
    def __init__(self, n_features, n_classes, hidden_sizes: List[int], activation='relu', dropout_rate:float=0.0):
        super(MlpClassifier, self).__init__()
        self.input_fc = Dense(hidden_sizes[0], activation=activation)
        self.input_dropout = Dropout(dropout_rate)
        self.n_features = n_features
        
        self.n_fc = len(hidden_sizes)-1
        for i, hidden_size in enumerate(hidden_sizes[1:]):
            fc = Dense(hidden_size, activation=activation)
            setattr(self, f'dense{i}', fc)
            setattr(self, f'dropout{i}', Dropout(dropout_rate))
        
        self.output_fc = Dense(n_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.input_fc(inputs)
        if training:
            x = self.input_dropout(x)
        for i in range(self.n_fc):
            fc = getattr(self, f'dense{i}')
            x = fc(x)
            if training:
                dropout = getattr(self, f'dropout{i}')
                x = dropout(x)
        return self.output_fc(x)
    
    def summary(self, training=True):
        x = Input(shape=(self.n_features,))
        Model(inputs=[x], outputs=self.call(x, training=training)).summary(print_fn=logger.info)
        
class MlpBinaryClassifier(tf.keras.Model):
    def __init__(self, n_features, hidden_sizes: List[int], activation='relu', dropout_rate:float=0.0):
        super(MlpBinaryClassifier, self).__init__()
        self.input_fc = Dense(hidden_sizes[0], activation=activation)
        self.input_dropout = Dropout(dropout_rate)
        self.n_features = n_features
        
        self.n_fc = len(hidden_sizes)-1
        for i, hidden_size in enumerate(hidden_sizes[1:]):
            fc = Dense(hidden_size, activation=activation)
            setattr(self, f'dense{i}', fc)
            setattr(self, f'dropout{i}', Dropout(dropout_rate))
        
        self.output_fc = Dense(1, activation='sigmoid')
    
    def call(self, inputs, training=False):
        x = self.input_fc(inputs)
        if training:
            x = self.input_dropout(x)
        for i in range(self.n_fc):
            fc = getattr(self, f'dense{i}')
            x = fc(x)
            if training:
                dropout = getattr(self, f'dropout{i}')
                x = dropout(x)
        return self.output_fc(x)
    
    def summary(self, training=True):
        x = Input(shape=(self.n_features,))
        Model(inputs=[x], outputs=self.call(x, training=training)).summary(print_fn=logger.info)
# model = MlpClassifier()
# predictions = model(input)
# model.fit(x, y, epochs=10, batch_size=32)



