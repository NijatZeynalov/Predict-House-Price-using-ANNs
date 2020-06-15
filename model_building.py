#import libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  MinMaxScaler


#build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(100, activation='relu', input_shape=(7, )))
model.add(tf.keras.layers.Dense(100, activation= 'relu'))
model.add(tf.keras.layers.Dense(100, activation= 'relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(1, activation='linear'))

# compile model
adam = tf.keras.optimizers.Adam(lr=0.002)
model.compile(optimizer = adam, loss = 'mean_squared_error')
epochs_hist = model.fit(X_train, y_train, epochs = 100, batch_size = 50, validation_split = 0.3)
model.save('house_model.h5')

#evaluate model
plt.plot(epochs_hist.history['loss'])
plt.plot(epochs_hist.history['val_loss'])
plt.legend(['Training loss', 'Validation loss'])
plt.show()

#predict model
#  bedrooms, bathrooms, sqft_living, sqft_lot, floors, sqft_above, sqft_basement
X_test1 = np.array([4, 5, 9000, 4900, 4, 4000, 3500 ])
scaler1 = MinMaxScaler()
X_test1_scaled = scaler1.fit_transform(X_test1)

y_pred_1 = model.predict(X_test1_scaled)
y_pred_1 = scaler1.inverse_transform(y_pred_1)
print(y_pred_1)