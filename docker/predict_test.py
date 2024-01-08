import requests

url = 'http://0.0.0.0:8080/predict'
     
example = {
    'host_response_rate': [.98], # My grandparents respond 98% of the time
    'host_acceptance_rate': [ 1], # Right now they always accept new guests
    'host_total_listings_count': [ 4], # They have 4 units
    'latitude': [ 19.37137], # ...
    'longitude': [ -99.19327],
    'accommodates': [ 4],
    'bathroom_qty': [ 2],
    'bedrooms': [ 2],
    'beds': [ 3],
    'host_response_time':['within_a_day'],
    'host_is_superhost': [True],
    'host_verifications_work_email':[1],
    'host_has_profile_pic': [True],
    'neighbourhood_cleansed': ['alvaro_obregon'],
    'room_type': ['entire_home_apt'],
    'property_description': ['home/apt'],
    'bathroom_type':['private_bath'],
    'gym': [0],
    'pool': [0],
    'kitchen': [1],
    'parking': [1],
    'washer': [1],
    'dryer': [1],
    'conditioning': [0],
    'heating': [0],
    'workspace': [1],
    'tv': [1],
    'hair dryer': [0],
    'iron': [1],
    'hot tub': [0],
    'crib': [0],
    'bbq': [0],
    'fireplace': [0],
    'smoking': [0],
    'coffee maker': [1],
    'instant_bookable': ['True'],
}

response = requests.post(url, json=example).json()
print(response)