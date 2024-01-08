import json

import pandas as pd
import numpy as np
import tensorflow as tf
import keras

from flask import Flask, request, abort, jsonify

app = Flask('airbnb')
@app.route('/predict', methods=['POST'])
def predict():
    # READ MODEL:
    model = keras.models.load_model(f'model.keras',safe_mode=False)
    features = ['host_response_rate', 'host_acceptance_rate',
       'host_total_listings_count', 'latitude', 'longitude', 'accommodates',
       'bathroom_qty', 'bedrooms', 'beds', 'host_response_time_within_a_day',
       'host_response_time_within_a_few_hours',
       'host_response_time_within_an_hour', 'host_is_superhost_True',
       'host_verifications_work_email_1', 'host_has_profile_pic_True',
       'neighbourhood_cleansed_azcapotzalco',
       'neighbourhood_cleansed_benito_juarez',
       'neighbourhood_cleansed_coyoacan',
       'neighbourhood_cleansed_cuajimalpa_de_morelos',
       'neighbourhood_cleansed_cuauhtemoc',
       'neighbourhood_cleansed_gustavo_a._madero',
       'neighbourhood_cleansed_iztacalco', 'neighbourhood_cleansed_iztapalapa',
       'neighbourhood_cleansed_la_magdalena_contreras',
       'neighbourhood_cleansed_miguel_hidalgo',
       'neighbourhood_cleansed_milpa_alta', 'neighbourhood_cleansed_tlahuac',
       'neighbourhood_cleansed_tlalpan',
       'neighbourhood_cleansed_venustiano_carranza',
       'neighbourhood_cleansed_xochimilco', 'room_type_hotel_room',
       'room_type_private_room', 'room_type_shared_room',
       'property_description_barn', 'property_description_bed_and_breakfast',
       'property_description_boutique_hotel', 'property_description_bungalow',
       'property_description_cabin', 'property_description_campsite',
       'property_description_casa_particular', 'property_description_castle',
       'property_description_chalet', 'property_description_condo',
       'property_description_cottage', 'property_description_dome',
       'property_description_dorm', 'property_description_earthen_home',
       'property_description_farm_stay', 'property_description_guest_suite',
       'property_description_guesthouse', 'property_description_holiday_park',
       'property_description_home', 'property_description_home/apt',
       'property_description_hostel', 'property_description_hotel',
       'property_description_houseboat', 'property_description_hut',
       'property_description_in-law', 'property_description_loft',
       'property_description_nature_lodge', 'property_description_pension',
       'property_description_place', 'property_description_private_room',
       'property_description_ranch', 'property_description_rental_unit',
       'property_description_serviced_apartment',
       'property_description_shared_room',
       'property_description_shipping_container', 'property_description_tent',
       'property_description_tiny_home', 'property_description_tipi',
       'property_description_tower', 'property_description_townhouse',
       'property_description_treehouse', 'property_description_vacation_home',
       'property_description_villa', 'bathroom_type_private_bath',
       'bathroom_type_shared_bath', 'gym_1', 'pool_1', 'kitchen_1',
       'parking_1', 'washer_1', 'dryer_1', 'conditioning_1', 'heating_1',
       'workspace_1', 'tv_1', 'hair dryer_1', 'iron_1', 'hot tub_1', 'crib_1',
       'bbq_1', 'fireplace_1', 'smoking_1', 'coffee maker_1',
       'instant_bookable_True']
    numerical = [
            'host_response_rate', 'host_acceptance_rate', 'host_total_listings_count',
            'accommodates', 'bathroom_qty', 'bedrooms', 'beds',
            ]
    geo = ['latitude','longitude']
    categorical = [
                    'host_response_time','host_is_superhost', 
                    'host_verifications_work_email','host_has_profile_pic',
                    'neighbourhood_cleansed', 'room_type', 'property_description', 'bathroom_type', 
                    'gym','pool','kitchen','parking','washer','dryer','conditioning','heating','workspace',
                    'tv','hair dryer','iron','hot tub','crib','bbq','fireplace','smoking', 'coffee maker', 
                'instant_bookable',
                ]
    complete_categorical=[i for i in features if i not in geo+numerical] # With dummy
    print(complete_categorical)
    label=['log_price']

    # READ DATA:
    example = request.get_json()
    example_df = pd.DataFrame.from_dict(example)

    # PREPROCESS DATA:  
    example_df = pd.get_dummies(example_df, columns=categorical, drop_first=False)

    missing_cols = set(features) - set(example_df.columns)
    for col in missing_cols:
        example_df[col] = 0

    example_df=example_df.astype('float32')

    example_df_ordered = example_df[geo+numerical+complete_categorical]

    # MODEL PREDICTION:
    price_unit =  np.exp(model.predict(example_df_ordered))[0][0]
    
    print('Predicted base price for the unit: $', price_unit, 'MXN/night')

    result = {
        'price_unit': float(price_unit),
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)