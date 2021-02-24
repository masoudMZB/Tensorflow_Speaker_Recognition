import tensorflow as tf
from tensorflow import keras

loss_history = [] 
model_to_train = None

def residual_block(x, filters, conv_num=3, activation='relu'):
  #shortcut
  s = keras.layers.Conv1D(filters, 1, padding='same', )(x)
  for i in range(conv_num -1):
    x = keras.layers.Conv1D(filters, 3, padding="same")(x)
    x = keras.layers.Activation(activation)(x)
  x = keras.layers.Conv1D(filters, 3, padding='same')(x)
  x = keras.layers.Add()([x, s])
  x = keras.layers.Activation(activation)(x)
  return keras.layers.MaxPool1D(pool_size=2, strides=2)(x)


def build_model(input_shape, num_classes):
  inputs = keras.layers.Input(shape=input_shape, name='audio_input')
  x = residual_block(inputs, 16, 2)
  x = residual_block(x, 32, 2)
  x = residual_block(x, 64, 3)
  x = residual_block(x, 128, 3)
  x = residual_block(x, 128, 3)

  x = keras.layers.AveragePooling1D(pool_size=3, strides=3)(x)
  x = keras.layers.Flatten()(x)
  x = keras.layers.Dense(256, activation="relu")(x)
  x = keras.layers.Dense(128, activation="relu")(x)
  outputs = keras.layers.Dense(num_classes, activation="softmax", name="output")(x)

  return keras.models.Model(inputs=inputs, outputs=outputs)



def prepare_model_and_train(sample_rate, class_names, train_ds, EPOCHS, valid_ds):
  model = build_model((sample_rate // 2, 1), len(class_names))
  #TODO Metric and GradientTape : https://keras.io/api/metrics/
  model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

  # CallBacks 
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")  # https://keras.io/api/callbacks/tensorboard/
  model_save_filename = "model.h5"
  earlystopping_cb = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)
  mdlcheckpoint_cb = keras.callbacks.ModelCheckpoint( model_save_filename, monitor='val_accuracy', save_best_only=True )

  history = model.fit(
      train_ds,
      epochs=EPOCHS,
      validation_data=valid_ds,
      callbacks=[earlystopping_cb, mdlcheckpoint_cb],
  )

  return history, model



def model_image(model, dot_img_file : str = './train/model_1.png'):
  tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=False)


# simple_gradient_tape approach
def train_step(audio, label, model):
  
  optimizer = tf.keras.optimizers.Adam()
  loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
  
  with tf.GradientTape() as tape:
    predicted = model(audio, training=True)
    loss_value = loss_object(label, predicted)

  loss_history.append(loss_value.numpy().mean())
  grads = tape.gradient(loss_value, model.trainable_variables)
  optimizer.apply_gradients(zip(grads, model.trainable_variables))


def train_gt(epochs, sample_rate, class_names, train_ds):
  model = build_model((sample_rate // 2, 1), len(class_names))
  for epoch in range(epochs):
    for (batch, (audios, labels)) in enumerate(train_ds):
      
      train_step(audios, labels, model)
    print ('Epoch {} finished'.format(epoch))
  
  return model








