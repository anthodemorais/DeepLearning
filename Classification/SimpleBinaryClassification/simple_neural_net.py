# first neural network with keras tutorial
from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from keras.models import model_from_json

# load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',') # split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model
model.fit(X, y, validation_split=0.33, epochs=150, batch_size=10)

# ----- OR -----
# X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)
# model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10

# ----- OR -----
# use k-fold cross-validation (best for small problems and small data because it's too greedy but less biased)

# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype("int32")
# np.argmax(model.predict(X), axis=-1) # for multi-class (softmax)

# ----- OR -----

# make probability predictions with the model
# predictions = model.predict(X)
# round predictions
# rounded = [round(x[0]) for x in predictions]

for i in range(5):
    print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

# works the same way for yaml format

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))