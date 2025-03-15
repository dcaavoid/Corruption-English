import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# Custom Attention Layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='att_weight', shape=(input_shape[-1], input_shape[-1]), initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight(name='att_bias', shape=(input_shape[-1],), initializer='zeros', trainable=True)
        self.u = self.add_weight(name='context_vector', shape=(input_shape[-1],), initializer='glorot_uniform', trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, inputs):
        uit = tf.tanh(tf.tensordot(inputs, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.u, axes=1)
        a = tf.nn.softmax(ait, axis=1)
        a = tf.expand_dims(a, axis=-1)
        output = tf.reduce_sum(inputs * a, axis=1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

def load_train_data(filename):
    """
    Each line in train.txt is in the format:
      natural_sentence<TAB>corrupted_sentence
    We create two training examples per line:
      1. (natural, corrupted) with label 0 ("A")
      2. (corrupted, natural) with label 1 ("B")
    """
    texts_a, texts_b, labels = [], [], []
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue

            natural = parts[0].strip()
            corrupted = parts[1].strip()

            # Original order with natural sentence first (label 0)
            texts_a.append(natural)
            texts_b.append(corrupted)
            labels.append(0)

            # Swapped order with natural sentence second (label 1)
            texts_a.append(corrupted)
            texts_b.append(natural)
            labels.append(1)
    return texts_a, texts_b, np.array(labels)

def load_test_data(filename):
    """
    Each line in test.rand.txt is in the format:
      sentence1<TAB>sentence2
    """
    texts_a, texts_b = [], []
    with open(filename, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) != 2:
                continue
            
            texts_a.append(parts[0].strip())
            texts_b.append(parts[1].strip())
    return texts_a, texts_b

def preprocess_texts(texts, tokenizer, max_len):
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded

def build_model(max_words, max_len, embedding_dim=100, lstm_units=64, dropout_rate=0.5):
    input_a = Input(shape=(max_len,), name='input_a')
    input_b = Input(shape=(max_len,), name='input_b')
    embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)
    bilstm = Bidirectional(LSTM(lstm_units, return_sequences=True, dropout=dropout_rate))
    att_layer = AttentionLayer()
    
    # Encode sentence A
    x1 = embedding(input_a)
    x1 = bilstm(x1)
    encoded_a = att_layer(x1)
    
    # Encode sentence B using the same layers
    x2 = embedding(input_b)
    x2 = bilstm(x2)
    encoded_b = att_layer(x2)
    
    diff = Lambda(lambda tensors: tensors[0] - tensors[1])([encoded_a, encoded_b])
    dense = Dense(64, activation='relu')(diff)
    drop = Dropout(dropout_rate)(dense)
    output = Dense(2, activation='softmax')(drop)
    
    model = Model(inputs=[input_a, input_b], outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    max_words = 10000
    max_len = 50
    embedding_dim = 100
    lstm_units = 64
    dropout_rate = 0.5
    batch_size = 64
    epochs = 20

    # Load and preprocess training data
    train_texts_a, train_texts_b, train_labels = load_train_data("challenge-data/train.txt")
    all_train_texts = train_texts_a + train_texts_b
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(all_train_texts)
    X1_train = preprocess_texts(train_texts_a, tokenizer, max_len)
    X2_train = preprocess_texts(train_texts_b, tokenizer, max_len)
    
    # Build and train the model with a validation split and early stop
    model = build_model(max_words, max_len, embedding_dim, lstm_units, dropout_rate)
    model.summary()
    early_stop = EarlyStopping(monitor='val_loss', patience=1, restore_best_weights=True)
    model.fit([X1_train, X2_train], train_labels, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=[early_stop])
    
    test_texts_a, test_texts_b = load_test_data("challenge-data/test.rand.txt")
    X1_test = preprocess_texts(test_texts_a, tokenizer, max_len)
    X2_test = preprocess_texts(test_texts_b, tokenizer, max_len)
    predictions = model.predict([X1_test, X2_test])
    pred_labels = ['A' if pred[0] > pred[1] else 'B' for pred in predictions]
    with open("solution/part1.txt", "w") as f_out:
        for label in pred_labels:
            f_out.write(f"{label}\n")
    
    print("Predictions saved to part1.txt")

if __name__ == '__main__':
    main()