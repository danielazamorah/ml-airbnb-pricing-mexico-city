"""
NOTE: The training code of nbs/02_training.ipynb was slightly edited to handle multiple 
inputs compatibility with the SHAP library (for model explanations). 
We modified the acrhitecture of the NN to receive a single input and then slice it
for further preprocessing [reference of issues with multiple inputs with SHAP: 
https://github.com/shap/shap/issues/857#issuecomment-566058378 and
https://github.com/shap/shap/issues/1945].

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

tf.random.set_seed(42)

from datetime import datetime

# Params:
TIMESTAMP = datetime.now().strftime("%Y%m%d%H%M%S")

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

# Define the new feature columns (categorical with dummy)
complete_categorical=[i for i in data.columns if i not in geo+numerical+label]

# Split data into training and testing sets
train_size = 0.8
train_dataset = data.sample(frac=train_size, random_state=0)
test_dataset = data.drop(train_dataset.index)

# Split features from labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()
train_label = train_features.pop('log_price')
test_label = test_features.pop('log_price')

# Save features and labels
data_dir = f"../data/train/v_{TIMESTAMP}"
os.makedirs(data_dir, exist_ok=True)
train_dataset.to_pickle(data_dir+"/train_dataset.pkl")  

data_dir = f"../data/test/v_{TIMESTAMP}"
os.makedirs(data_dir, exist_ok=True)
test_dataset.to_pickle(data_dir+"/test_dataset.pkl")  

# Define input
def build_model():
    # Define the input layer
    input_tensor = tf.keras.layers.Input(shape=(train_features.shape[1],), dtype=tf.float32)

    # Lambda layers for slicing
    # Extracting latitude (assuming it's at a specific index, e.g., 0)
    latitude_layer = tf.keras.layers.Lambda(lambda x: x[:, 0:1])(input_tensor)

    # Extracting longitude (assuming it's at a specific index, e.g., 1)
    longitude_layer = tf.keras.layers.Lambda(lambda x: x[:, 1:2])(input_tensor)

    # Extracting numerical variables
    # Assuming numerical variables are at indices 2 to 2+len(numerical)-1
    numerical_layer = tf.keras.layers.Lambda(lambda x: x[:, 2:9])(input_tensor)

    # Extracting categorical variables
    # Assuming categorical variables start right after numerical
    categorical_layer = tf.keras.layers.Lambda(lambda x: x[:, 9:97])(input_tensor)

    # Preprocessing layers
    normalizer_rest = tf.keras.layers.Normalization(axis=-1)  # Normalization layer for numerical features
    numerical_features = train_features[numerical]
    normalizer_rest.adapt(np.array(numerical_features))  # Adapt normalization layer to training data

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

    # Discretization and encoding for latitude
    latitude = tf.keras.layers.Discretization(bin_boundaries=lat_boundaries, name='discretization_latitude')(latitude_layer)
    latitude = tf.keras.layers.CategoryEncoding(num_tokens=len(lat_boundaries) + 1, output_mode='one_hot',
                                                name='category_encoding_latitude')(latitude)

    # Discretization and encoding for longitude
    longitude = tf.keras.layers.Discretization(bin_boundaries=long_boundaries, name='discretization_longitude')(longitude_layer)
    longitude = tf.keras.layers.CategoryEncoding(num_tokens=len(long_boundaries) + 1, output_mode='one_hot',
                                                name='category_encoding_longitude')(longitude)

    # Preprocess numerical and categorical inputs
    numerical_layer_norm = normalizer_rest(numerical_layer)

    # Concatenate all preprocessed inputs
    concatenated_features = tf.keras.layers.Concatenate()([latitude, longitude, numerical_layer_norm, categorical_layer])

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
    model = tf.keras.Model(
        inputs=input_tensor, 
        outputs=output,
        )
    
    # Compile the model
    model.compile(
        optimizer=tf.keras.optimizers.legacy.RMSprop(learning_rate=0.001),
        loss="mean_squared_error",
        metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
    return model

model = build_model()

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
# Fit the model to the training data
history = model.fit(
    train_features[geo+numerical+complete_categorical],
    train_label,
    epochs=200,
    verbose=2,
    validation_split=0.2,
    callbacks=[stop_early],
)

# Evaluate the model on the test set
res = model.evaluate(
    test_features[geo+numerical+complete_categorical],
    y=test_label, verbose=0)

# Gather and display test metrics
metrics = model.metrics_names
test_results = {metrics[0]: res[0], metrics[1]: res[1]}
print(' TEST RESULTS: ',test_results)

# Creating the directory path for the model
model_dir = f"../models/v_{TIMESTAMP}"

# Create the directory if it does not exist
os.makedirs(model_dir, exist_ok=True)

# Saving the model
model_path = os.path.join(model_dir, "model.keras")
model.save(model_path)