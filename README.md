**Bina.az Rent Price Prediction**

This project focuses on scraping rental apartment listings from bina.az and building a machine learning regression model to predict monthly rental prices based on apartment characteristics such as room count, area, floor, and other relevant features.

The project demonstrates an end-to-end machine learning workflow, from data collection to preprocessing, feature engineering, model training, and evaluation.

Data Collection
Dataset Overview

The data is collected from bina.az. The dataset consists of the following features:

**Rooms**: Number of rooms in the apartment

**Area (m²)**: Area of the apartment in square meters

**Floor Current**: Current floor of the apartment

**Floor Total**: Total number of floors in the building

**Location**: The city and district of the apartment

**Price**: The monthly rental price (target variable)

Data was collected using **Selenium** and **BeautifulSoup**, with infinite scrolling on the website, to dynamically load additional listings.

**Maximum Listings**: 15,000
Only monthly rental listings (/ay) were included in the dataset. Daily listings (/gün) were excluded intentionally to ensure consistent target variable predictions.

Data Preprocessing
1. Location Cleaning

The location column was split into city and district columns, which helps the model capture location-specific price effects.

df['location_clean'] = df['location'].apply(lambda x: x.split(' - ')[-1])
df[['city', 'district']] = df['location_clean'].str.split(',', n=1, expand=True)
df['city'] = df['city'].str.strip()
df['district'] = df['district'].str.strip()
df['district'] = df['district'].apply(lambda x: x.split(',')[0].strip())

**2. Outlier Filtering**

The dataset was filtered using the 1% - 99% quantiles for price, area, and room values to remove outliers.

Price, Area, and Rooms were bounded within reasonable ranges to remove extreme outliers.

price_min, price_max = df['price'].quantile([0.01, 0.99])
area_min, area_max = df['area_m2'].quantile([0.01, 0.99])
rooms_min, rooms_max = df['rooms'].quantile([0.01, 0.99])

df_clean = df[
    df['price'].between(price_min, price_max) &
    df['area_m2'].between(area_min, area_max) &
    df['rooms'].between(rooms_min, rooms_max) &
    (df['floor_current'] <= df['floor_total'])
].copy()

**3. Feature Engineering**

**Building Type**: Building type was inferred based on the total number of floors: buildings with more than 11 floors were considered New buildings, while others were considered Old.

**Floor Ratio**: The ratio of current floor to total floor count (floor_ratio).

**City Encoding**: The city column was transformed into a binary variable, where Baku is 1, and other cities are 0 (is_baku).

**District Target Encoding**: The district column was encoded using target encoding, where the mean rental price for each district is used as the encoded value.

**4. Dataset Files**

**bina_rent_clean.csv**: Contains cleaned data (location, rooms, area, etc.).

**bina_rent_model.csv**: Contains data after feature engineering, including encoded features (district_target_enc, is_baku, etc.).

df_clean.to_csv("../data/processed/bina_rent_clean.csv", index=False)
df_model.to_csv("../data/processed/bina_rent_model.csv", index=False)

**Modeling**
**1. Target and Features**

The goal is to predict the price of apartments. The feature columns (X) include:

rooms, area_m2, floor_current, floor_total, floor_ratio, building_type_enc, is_baku, district_freq, district_target_enc

The target column (y):

**price**

X = df_model.drop(columns=["price"])
y = df_model["price"]

**2. Train / Test Split**

The dataset is split into train and test sets, with a test size of 20%.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

**3. District Target Encoding (train/test split)**

Target Encoding was applied to the district feature based on the mean rental price for each district.

Train/test split was used to ensure no data leakage. Target encoding is done only on the train set, and the test set uses the fillna method.

district_mean = y_train.groupby(X_train["district"]).mean()
X_train["district_target_enc"] = X_train["district"].map(district_mean)
X_test["district_target_enc"] = X_test["district"].map(district_mean)

**4. Model Evaluation**

Model performance was evaluated using **MAE, RMSE, and R²** metrics. Results were recorded for both train and test sets.

eval_metrics(y_train, y_pred_train, "Train XGB")
eval_metrics(y_test, y_pred_test, "Test XGB")

Model Evaluation

After evaluating various models, including Random Forest and MLP, the best-performing model was chosen based on the following results:

**Train Model** → MAE: 91.54 | RMSE: 134.23 | R²: 0.877

**Test Model** → MAE: 130.31 | RMSE: 196.24 | R²: 0.735

These results are from the XGBoost model, which provided the best performance among the models tested.

**Additional Notes**

**Data Limit**: The maximum number of listings scraped was set to 15,000.

**Location Filtering**: Only monthly rental listings were included in the dataset, excluding daily rental listings.

**Target Encoding**: Target Encoding was only performed on the train set to prevent data leakage. The test set was encoded separately.

**Future Improvements**

Further tuning of the hyperparameters using techniques like GridSearchCV or Optuna can potentially improve model performance.

Additional models such as Random Forest or MLP can be tested and compared further.

The model can be trained on additional features, such as renovation status or building age, if more data becomes available.

**Additional Notes**

This project demonstrates the process of predicting apartment rental prices using a machine learning model, specifically focusing on preprocessing, feature engineering, and evaluation of various algorithms.

More advanced techniques like ensemble methods or neural networks could be used in the future to improve prediction accuracy.

**Note:**

Make sure to update the file paths for your system and adjust any hyperparameters if you make changes. For example, the price filter or the target encoding strategy might change depending on the final data you're working with.

## Conclusion

This project demonstrates how to build a machine learning model to predict apartment rental prices. It covers everything from data collection and preprocessing to model training and evaluation.

Initially, the **XGBoost** model was chosen for prediction, but after further experimentation, the **Random Forest** model delivered better results in terms of accuracy and performance. Future improvements and refinements to the model could potentially increase prediction accuracy further.

The Random Forest model, with its ability to handle non-linear relationships and interactions between features, has proven to be more suitable for this dataset. Further hyperparameter tuning, as well as testing with additional features, could enhance model performance even more.

Feel free to explore the code and contribute if you have ideas for improving the model!

