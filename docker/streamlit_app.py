import datetime
import streamlit as st
import requests
import json
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
latitude = st.number_input("Latitude",step=.001,format="%.5f")
longitude = st.number_input("Longitude",step=.001,format="%.5f")
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

    url = 'http://flask-app:8080/predict'

    try:
        response = requests.post(url, json=example).json()
        price_unit = response['price_unit']
        st.write("Predicted Price: $%.2f MXN/night"%price_unit)
        st.write("Try adding amenities or changing other variables to see how it affects the price!")
    except ValueError:
        st.error('Please provide all the charactherisics of the Airbnb host/unit')