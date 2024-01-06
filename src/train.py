"""
This script is for building and training a deep neural network (DNN) model for regression, 
specifically designed to predict the log-transformed price (log_price) based on various 
features of rental listings. 

It includes preprocessing steps such as one-hot encoding of categorical variables, 
normalization of numerical variables, and discretization and encoding of geographical 
features (latitude and longitude). The model uses TensorFlow and Keras APIs for building, 
compiling, and training the DNN.
"""
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Define the feature columns
numerical = ['host_response_rate', 'host_acceptance_rate', 'host_total_listings_count',
             'accommodates', 'bathroom_qty', 'bedrooms', 'beds']
geo = ['latitude', 'longitude']
categorical = ['host_response_time', 'host_is_superhost', 
               'host_verifications_work_email', 'host_has_profile_pic',
               'neighbourhood_cleansed', 'room_type', 'property_description', 
               'bathroom_type', 'gym', 'pool', 'kitchen', 'parking', 'washer', 
               'dryer', 'conditioning', 'heating', 'workspace', 'tv', 
               'hair dryer', 'iron', 'hot tub', 'crib', 'bbq', 'fireplace', 
               'smoking', 'coffee maker', 'instant_bookable']
label = ['log_price']

# Load and preprocess data
data = pd.read_pickle("../data/listings_cleaned.pkl")
data.loc[:, "log_price"] = np.log(data.loc[:, "price"])  # Convert price to log scale for normalization
data = data[numerical + categorical + geo + label]  # Combine all features and the label

# Convert categorical features to one-hot encoding
data = pd.get_dummies(data, columns=categorical, drop_first=True)
data = data.astype('float32')  # Ensure all data is in float32 for TensorFlow compatibility

# Split data into training and testing sets
train_size = 0.8
train_dataset = data.sample(frac=train_size, random_state=0)
test_dataset = data.drop(train_dataset.index)

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_label = train_features.pop('log_price')
test_label = test_features.pop('log_price')

# Create bins for longitude
max_long = train_dataset['longitude'].max()
min_long = train_dataset['longitude'].min()
diff = (max_long - min_long) / 100
long_boundaries = [min_long + i * diff for i in np.arange(min_long, max_long, diff)]

# Create bins for latitude
max_lat = train_dataset['latitude'].max()
min_lat = train_dataset['latitude'].min()
d = (max_lat - min_lat) / 100
lat_boundaries = [min_lat + i * d for i in np.arange(min_lat, max_lat, d)]

# Model configuration and training
rmse = tf.keras.metrics.RootMeanSquaredError()

# Define the input layers for each feature
inputs_all = {
    'latitude': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='latitude'),
    'longitude': tf.keras.layers.Input(shape=(1,), dtype=tf.float32, name='longitude'),
    'other_numerical': tf.keras.layers.Input(shape=(len(numerical),), name='other_features_numerical'),
    'other_categorical': tf.keras.layers.Input(shape=(train_features.shape[1] - len(geo) - len(numerical),), 
                                               name='other_features_categorical')
}

# Preprocessing layers
normalizer_rest = tf.keras.layers.Normalization(axis=-1)  # Normalization layer for numerical features
numerical_features = train_features[numerical]
normalizer_rest.adapt(np.array(numerical_features))  # Adapt normalization layer to training data

# Discretization and encoding for latitude
latitude = tf.keras.layers.Discretization(bin_boundaries=lat_boundaries, name='discretization_latitude')(inputs_all['latitude'])
latitude = tf.keras.layers.CategoryEncoding(num_tokens=len(lat_boundaries) + 1, output_mode='one_hot',
                                            name='category_encoding_latitude')(latitude)

# Discretization and encoding for longitude
longitude = tf.keras.layers.Discretization(bin_boundaries=long_boundaries, name='discretization_longitude')(inputs_all['longitude'])
longitude = tf.keras.layers.CategoryEncoding(num_tokens=len(long_boundaries) + 1, output_mode='one_hot',
                                             name='category_encoding_longitude')(longitude)

# Preprocess numerical and categorical inputs
other_features_numerical = normalizer_rest(inputs_all['other_numerical'])
other_features_categorical = inputs_all['other_categorical']

# Concatenate all preprocessed inputs
concatenated_features = tf.keras.layers.Concatenate()([latitude, longitude, other_features_numerical, other_features_categorical])

# Define and build the neural network architecture
x = layers.Dense(384, activation='relu')(concatenated_features) # layer 1
x = layers.Dropout(0.2)(x)
x = layers.Dense(224, activation='relu')(x) # layer 2
x = layers.Dropout(0.1)(x) 
x = layers.Dense(480, activation='relu')(x) # layer 3
x = layers.Dropout(0.1)(x)
x = layers.Dense(64, activation='relu')(x) # layer 4
x = layers.Dropout(0.1)(x)
x = layers.Dense(512, activation='relu')(x) # layer 5
x = layers.Dropout(0.2)(x)
output = layers.Dense(1)(x)  # Output layer for regression

# Create the model
model = tf.keras.Model(inputs=inputs_all, outputs=output)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
              loss="mean_squared_error",
              metrics=[tf.keras.metrics.RootMeanSquaredError()])

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Fit the model to the training data
history = model.fit(
    {
        'latitude': train_features['latitude'],
        'longitude': train_features['longitude'],
        'other_numerical': numerical_features,
        'other_categorical': train_features.drop(geo + numerical, axis=1),
    },
    train_label,
    epochs=200,
    verbose=2,
    validation_split=0.2,
    callbacks=[stop_early],
)

# Evaluate the model on the test set
res = model.evaluate(
    x={
        'latitude': test_features['latitude'],
        'longitude': test_features['longitude'],
        'other_numerical': test_features[numerical],
        'other_categorical': test_features.drop(['latitude', 'longitude'] + numerical, axis=1),
    },
    y=test_label, verbose=0)

# Gather and display test metrics
metrics = model.metrics_names
test_results = {metrics[0]: res[0], metrics[1]: res[1]}