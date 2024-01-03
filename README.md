# Machine Learning Airbnb Pricing in Mexico City! 

Project where we create and deploy a TensorFlow Machine Learning model for pricing recommendations on Mexico City's new Airbnb listings!

## Inspiration story

My grandparents placed a property on Airbnb. How did they select the price? Looking at the houses in the area and selecting a lower base price to be competitive (not taking into account rooms, amenities, descriptions, types, etc🤔). Airbnb does recommend a base price when creating a new property, but they don't give a lot of explanation on the factors that influence that recommendation!

<img src="assets/images/out_new_suggested_price.png">

You can only check the map with the nearest listings.

<img src="assets/images/out_new_area_prices.png">

And they mention the following:

> To determine listings that are similar to yours, we consider criteria like location, listing type, rooms, amenities, reviews, ratings, and the listings that guests often view alongside yours.

So, in this project, our main objective will be obtaining the price and explaining where the prediction comes from. Which factors could my grandparents change on their units to make their base price per night better?

## Set Environment

To run the notebooks, training, and deployment pieces of the project:
1. Clone the repository:
    ```
    git clone https://github.com/datasciencedani/ml-airbnb-pricing-mexico-city.git
    ```

1. Ensure environment with `virtualenv`:
    ```
    virtualenv venv
    ```
    ```
    source venv/bin/activate
    ```
    ```
    pip install -r requirements.txt
    ```
1. Create a jupyter kernell for the environment:
    ```
    python -m ipykernel install --user --name=env-ml-airbnb
    ```

## Run yourself

The first thing you can run is the notebooks where we prepare our data for modeling:
1. [Clean the data NB](nbs/00_cleaning.ipynb): correct data types, extract variables from raw data and select features to analyze.
1. [Exploratory Data Analysis (EDA) NB](nbs/01_eda.ipynb): observe the relationship between our target variable (price) and the features we selected previously (Airbnb host and listing features) to select the initial set of variables we'll use to model price.
1. [Data preparation NB](nbs/02_data_prep.ipynb): where we prepare the code to have a dataset ready for our TensorFlow modeling.

## Research and prepwork

- The data we use: http://insideairbnb.com/get-the-data

- Inspiration project: https://towardsdatascience.com/data-cleaning-and-eda-on-airbnb-dataset-with-python-pandas-and-seaborn-7c276116b650

- Project already using tensorflow: https://github.com/Timothy102/Tensorflow-for-Airbnb-Prices/blob/main/Rentals.ipynb

- Pricing strategy by Airbnb: https://www.airbnb.com/resources/hosting-homes/a/how-to-set-a-pricing-strategy-15

- Cool idea on second iteration: use also images of the listings as determinants for the price.

- Color codes airbnb: https://usbrandcolors.com/airbnb-colors/