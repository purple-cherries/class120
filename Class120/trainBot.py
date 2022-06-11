from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import Adam
from data_preprocessing import preprocess_train_data

def train_bot_model(trainx, trainy):
    model = Sequential()
    model.add(Dense(128, input_shape = (len(trainx[0]),), activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(trainy[0]), activation = 'relu'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    history = model.fit(trainx, trainy, epochs = 200, batch_size = 5, verbose = True)
    model.save('ChatbotModel.h5', history)

trainx, trainy = preprocess_train_data()
train_bot_model(trainx, trainy)
