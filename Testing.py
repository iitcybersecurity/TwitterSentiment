import os
import pickle
from keras import models
import settings

def predict_tweet(tweet_text):

    MAX_SEQUENCE_LENGTH = 30
    # preprocessing step (tweet conversion using word embedding)
    # loading
    with open(os.path.join(settings.HOME_DIRECTORY, 'tokenizer.pickle'), 'rb') as handle:
        tokenizer = pickle.load(handle)
    word_index = tokenizer.word_index
    vocab_size = len(tokenizer.word_index) + 1
    print("Vocabulary Size :", vocab_size)

    from keras.preprocessing.sequence import pad_sequences

    x_test = pad_sequences(tokenizer.texts_to_sequences(tweet_text),
                           maxlen=MAX_SEQUENCE_LENGTH)

    model = models.load_model(os.path.join(settings.HOME_DIRECTORY, "model_network"))
    model.load_weights(os.path.join(os.path.join(settings.HOME_DIRECTORY, 'checkpoint'), 'weights.ckpt'))

    predictions = model.predict(x_test)
    print(predictions)

predict_tweet(['today is a good day', 'this is horrible'])