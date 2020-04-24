from model import *
from tensorflow.keras import callbacks

model = build_model()

log_dir = '/home/aaron/PycharmProjects/US_Model/logs'
fp = '/home/aaron/PycharmProjects/US_Model/models/'
labels_tsv = '/home/aaron/PycharmProjects/US_Model/labels.tsv'

callbacks = [callbacks.ModelCheckpoint(filepath=fp+'coronavirus_US.hdf5', verbose=1),
             callbacks.TensorBoard(log_dir=log_dir, write_images=False, embeddings_freq=1)]

model.compile('adam', loss='mse', metrics=['accuracy'])

model.fit(training_dataset, batch_size=None, epochs=10000, verbose=1, callbacks=callbacks,
          steps_per_epoch=2500, validation_data=test_dataset, validation_freq=2, validation_steps=500)
