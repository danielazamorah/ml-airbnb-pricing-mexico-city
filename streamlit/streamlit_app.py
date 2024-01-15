import os
import datetime
import json
import requests

import streamlit as st
import pandas as pd
import numpy as np

from google.cloud import aiplatform
from google.oauth2 import service_account
###############################################################
st.title('Mexico City Airbnb Price Predictor!')
st.write('### Experiment with different characteristics of your AirBnb unit 👩🏽‍🔬')
st.write('What could you offer that would improve your price? Remember that the model behind this tool was trained using AirBnb data from the last quarter of 2023 on Mexico City.')

###############################################################
st.header("Host Information")
host_response_rate = st.slider("Response Rate", 0.0, 100.0, 100.0)
host_acceptance_rate = st.slider("Acceptance Rate", 0.0, 100.0, 100.0)
total_listings = st.number_input("Number of Airbnb listings", min_value=1)
response_time = st.selectbox("Host's Typical Response Time", 
                             ['within_a_day','within_a_few_hours','within_an_hour','a_few_days_or_more'])
is_superhost = st.checkbox("Super Host")
work_email_verified = st.checkbox("Work Email Verified", value=False)
has_profile_pic = st.checkbox("Has Profile Picture")
###############################################################
st.header("Unit's Information")

st.subheader("Accommodation")
accommodates = st.number_input("Accommodates", min_value=1)
bathrooms = st.number_input("Bathrooms", min_value=1)
bedrooms = st.number_input("Bedrooms", min_value=0)
bathroom_type = st.selectbox("Bathroom Type", ['bath', 'private_bath', 'shared_bath'])
beds = st.number_input("Beds", min_value=0)

st.subheader("Location")
latitude = st.number_input("Latitude",step=.001,format="%.5f", value=19.4326)
longitude = st.number_input("Longitude",step=.001,format="%.5f", value=99.1332)
neighbourhood = st.selectbox("Neighbourhood", ['azcapotzalco','benito_juarez','coyoacan','cuajimalpa_de_morelos', 'cuauhtemoc', 'gustavo_a._madero', 'iztacalco', 'iztapalapa', 'la_magdalena_contreras', 'miguel_hidalgo', 'milpa_alta', 'tlahuac', 'tlalpan',  'venustiano_carranza', 'xochimilco', 'alvaro_obregon'])  # Add all options

st.subheader("Description")
room_type = st.selectbox("Room Type", ["entire_home_apt", "private_room", "hotel_room", "shared_room"])  # Add all options
property_description = st.selectbox("Property Description", ['barn', 'bed_and_breakfast','boutique_hotel', 'bungalow','cabin', 'campsite','casa_particular','castle','chalet', 'condo','cottage', 'dome','dorm', 'earthen_home','farm_stay', 'guest_suite','guesthouse', 'holiday_park','home', 'home/apt','hostel', 'hotel','houseboat', 'hut','in-law', 'loft','nature_lodge', 'pension','place', 'private_room','ranch', 'rental_unit','serviced_apartment','shared_room','shipping_container', 'tent','tiny_home', 'tipi','tower', 'townhouse','treehouse', 'vacation_home','villa'])  # Add all options
instant_bookable = st.checkbox("Instantly Bookable")

st.subheader("Amenities")
amenities = {amenity: st.checkbox(amenity.replace('_', ' ').title()) for amenity in 
             ['gym','pool','kitchen','parking','washer','dryer','conditioning','heating','workspace', 'tv','hair_dryer','iron','hot_tub','crib','bbq','fireplace','smoking', 'coffee_maker']}  # Add all amenities
###############################################################

if st.button("Predict Price"):
    example = {
    'host_response_rate': [host_response_rate / 100], 
    'host_acceptance_rate': [host_acceptance_rate / 100], 
    'host_total_listings_count': [total_listings], 
    'latitude': [latitude],
    'longitude': [longitude],
    'accommodates': [accommodates],
    'bathroom_qty': [bathrooms],
    'bedrooms': [bedrooms],
    'beds': [beds],
    'host_response_time':[response_time],
    'host_is_superhost': [bool(is_superhost)],
    'host_verifications_work_email':[int(work_email_verified)], 
    'host_has_profile_pic': [bool(has_profile_pic)],
    'neighbourhood_cleansed': [neighbourhood],
    'room_type': [room_type],
    'property_description': [property_description],
    'bathroom_type':[bathroom_type],
    'gym': [int(amenities['gym'])],
    'pool': [int(amenities['pool'])],
    'kitchen': [int(amenities['kitchen'])],
    'parking': [int(amenities['parking'])],
    'washer': [int(amenities['washer'])],
    'dryer': [int(amenities['dryer'])],
    'conditioning': [int(amenities['conditioning'])],
    'heating': [int(amenities['heating'])],
    'workspace': [int(amenities['workspace'])],
    'tv': [int(amenities['tv'])],
    'hair dryer': [int(amenities['hair_dryer'])],
    'iron': [int(amenities['iron'])],
    'hot tub': [int(amenities['hot_tub'])],
    'crib': [int(amenities['crib'])],
    'bbq': [int(amenities['bbq'])],
    'fireplace': [int(amenities['fireplace'])],
    'smoking': [int(amenities['smoking'])],
    'coffee maker': [int(amenities['coffee_maker'])],
    'instant_bookable': [bool(instant_bookable)],
    }

    # ------------------------------------------------------------------
    # ------------------------------------- Prepare example for request:
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
    label=['log_price']
    complete_categorical=[i for i in features if i not in geo+numerical]

    example_df = pd.DataFrame.from_dict(example)

    # 1. Apply pd.get_dummies to the example_df
    example_df = pd.get_dummies(example_df, columns=categorical, drop_first=False)

    # 2. . Add missing columns to example_df
    missing_cols = set(features) - set(example_df.columns)
    for col in missing_cols:
        example_df[col] = 0

    # 3. Ensure the order of columns is the same as in 'data'
    example_df = example_df[features]

    # 4. Reformat
    example_df=example_df.astype('float32')

    # 5. Order for predictions:
    test_features_ordered = example_df[geo+numerical+complete_categorical]

    # ------------------------------------------------------------------
    # ----------------------------------------- Send request to endpoint:
    #try:
    credentials = service_account.Credentials.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    endpoint_id = '161753553608638464'
    project="532579765435"
    location="us-central1"

    endpoint_name = f'projects/{project}/locations/{location}/endpoints/{endpoint_id}'
    endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name, credentials=credentials)

    price_unit = np.exp(endpoint.predict(instances=[test_features_ordered.values[0].astype(np.float32).tolist()]).predictions[0][0])

    st.write("Predicted Price: $%.2f MXN/night"%price_unit)
    st.write("Try adding amenities or changing other variables to see how it affects the price!")
    #except ValueError:
    #    st.error('Please provide all the charactherisics of the Airbnb host/unit. If you have, then there is an error with endpoint connection. Please try again later.')