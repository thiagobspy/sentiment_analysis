import numpy as np
import keras

from keras.datasets import imdb
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding


top_words = 5000

(X_train, Y_train),(X_test, Y_test) = imdb.load_data(num_words=top_words)

NUM_WORDS=1000 # only use top 1000 words
INDEX_FROM=3   # word index offset
word_to_id = keras.datasets.imdb.get_word_index()
word_to_id = {k:(v+INDEX_FROM) for k,v in word_to_id.items()}
word_to_id["<PAD>"] = 0
word_to_id["<START>"] = 1
word_to_id["<UNK>"] = 2

id_to_word = {value:key for key,value in word_to_id.items()}
print(' '.join(id_to_word[id] for id in X_train[0] ))

#Transformar um documento com palavras em indices
docs = ["Thiago Soares", "Thiago Batista", "Blusa Fria", "Thiago Blusa Fria Soares"]
vocab_size = 10
encoded_docs = [one_hot(d, vocab_size) for d in docs]
# print(encoded_docs)


max_review_length = 500

X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

embedding_vector_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vector_length, input_length=max_review_length))
model.add(LSTM(100))
#model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
print(model.summary())
        

model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=3, batch_size=64)

score = model.evaluate(X_test, Y_test, verbose=0)
print("Accuracy: %.2f%%" % (score[1]*100))

bad = "this movie was terrible and bad"
good = "i really liked the movie and had fun"
bad_2 = "i do not like this movie"
good_2 = "the best movie"
bad_3 = "i need to die after i saw this movie"
good_3 = "please i need see again"

for review in [bad, good, bad_2, good_2, bad_3, good_3]:
    tmp = []
    for word in review.split(" "):
        tmp.append(word_to_id[word])
    tmp_padded = sequence.pad_sequences([tmp], maxlen=max_review_length)
    print("%s . Sentiment: %s" % (review,model.predict(np.array([tmp_padded][0]))[0][0]))
