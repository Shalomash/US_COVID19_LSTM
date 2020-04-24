from data_preprocessing import preprocess_data
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model


sequence_length = 30
buffer_size = 1024
batch_size = 32

training_dataset, test_dataset, data_std, data_mean, n_state_labels, n_county_labels = preprocess_data(sequence_length=sequence_length, buffer_size=buffer_size,
                                                        batch_size=batch_size)


def build_model():

    state_emb_input = Input(shape=(1,))
    county_emb_input = Input(shape=(1,))
    data_input = Input(shape=(sequence_length, 3))

    state_emb = Embedding(input_dim=n_state_labels, output_dim=8, input_length=1)(state_emb_input)
    county_emb = Embedding(input_dim=n_county_labels, output_dim=8, input_length=1)(county_emb_input)
    embedding = Concatenate()([state_emb, county_emb])
    embedding = Flatten()(embedding)
    embedding = Dense(sequence_length, activation='linear')(embedding)
    embedding = Reshape((sequence_length, 1))(embedding)
    concat = Concatenate()([data_input, embedding])

    out = LSTM(128, dropout=0.2, return_sequences=True)(concat)
    out = LSTM(256, dropout=0.2, return_sequences=False)(out)
    out1 = Dense(1, activation='linear', name='Confirmed_Cases')(out)
    out2 = Dense(1, activation='linear', name='Total_Deaths')(out)

    model = Model([state_emb_input, county_emb_input, data_input], [out1, out2])
    model.build(input_shape=((1,), (1,), (sequence_length, 1)))
    model.summary()
    return model