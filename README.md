# nyc-taxi-fare-prediction usign pytorch  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/farazabir/nyc-taxi-fare-prediction/blob/main/nyc_taxi_fare_tabularmodel.ipynb)
Predicting taxi fares based on tabular data and using PyTorch for building and training a neural network using both categorical and continuous data.


The dataset used contains taxi ride details such as:
- `pickup_datetime`: Timestamp of ride start.
- `fare_amount`: Fare charged for the ride.
- `fare_class`: Categorical representation of fare range.
- `pickup_longitude`, `pickup_latitude`: Coordinates of ride start.
- `dropoff_longitude`, `dropoff_latitude`: Coordinates of ride end.
- `passenger_count`: Number of passengers in the ride.


## Preprocessing Steps

1. **Feature Engineering**: Computed haversine distance (`dist_km`) for geographical distance between pickup and dropoff points.
2. **Datetime Features**: Extracted features like hour of the day, day of the week, and AM/PM from the `pickup_datetime`.
3. **Categorical Encoding**: Used PyTorch Embeddings for categorical features: `Hour`, `AM/PM`, `Weekday`.
4. **Normalization**: Applied batch normalization on continuous features.


