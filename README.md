# Car Auction Predictive Analytics Report
---


# Data Imputation Framework vs. Row Dropping Strategy

After importing the dataset, we ensured the removal of rows containing **3 or more missing columns**. Following this:

* Categorical columns were converted to lowercase.
* Unnecessary gaps and spaces were removed.
* The `make` feature was trimmed to its first 4 letters.

### Missing Value Imputation Strategy

#### Model Imputation

The missing `model` values were imputed using the **mode of `make` and `year`**, i.e., the most frequently occurring model for a particular make in a given year.

#### Trim Imputation

The missing `trim` values were imputed using related attributes such as:

* `trim`
* `model`
* `body`

#### Color and Interior Imputation

* `color` was imputed using `year` and `body`.
* `interior` was imputed using `exterior`.

### Imputing Missing Odometer Values

Odometer values are highly dependent on the manufacturing year. Therefore, we used **Linear Regression** to fit and predict missing odometer values with respect to the `year` feature.

---

# Some Categorical Feature Engineering

## Trim Tier

In this step, we classified trims into the following categories using appropriate keywords:

* Luxury
* Sport
* Mid-High
* Mid
* Base

## Broad Body

Similarly, vehicle body types were classified into:

* SUV
* Sedan
* Coupe
* Wagon
* Convertible
* Truck
* Cab
* Van
* Hatchback
* Other

## Color Demand

Vehicle colors were classified based on demand levels:

* High
* Medium
* Niche
* Rare

Similarly, interior masking was classified into:

* High
* Mid
* Low
* Other

---

# Encoded Features

We created encoded features for all the above categorical columns using **Category Encoders**.

The encoding process used a target column (here, the `selling price`) and encoded values based on the **median selling price** of a particular feature.

---

# Further Imputation of Transmission and Condition

For these features, we used machine learning models to predict missing values.

* **LGBMClassifier** was used to predict the `transmission`.
* **LGBMRegressor** was used to predict the `condition`.

---

# Outlier Detection and Removal

## Odometer Values

We used the **IQR (Interquartile Range) Method** to detect outliers in odometer values.

Since odometer readings are highly dependent on the manufacturing year, outlier detection was performed **separately for each year**.

### Outlier Handling

Outliers were removed by replacing them with the **median value of the respective year**.

The threshold for outlier detection was:

[
1.5/times IQR
]

# Selling Price

Similarly, we used a **2-Step Outlier Detection Method** for identifying anomalies in selling prices.

1. We flagged outliers with respect to selling price along with `make`, `model`, and `year`.
2. For handling these outliers, we used **LightGBM Regressor** to resolve abnormal values.

---

# Some Other Numerical Feature Engineering

We created features such as:

* `age`
* `usage density`
* `condition ratio`

These were engineered using:

* `year`
* `odometer`
* `condition`

values.

---

# Exploratory Data Analysis (EDA)

After feature engineering, we performed detailed **Exploratory Data Analysis (EDA)** to understand:

* Vehicle pricing trends
* Depreciation behavior
* Mileage influence
* Brand-wise distribution
* Regional market patterns
* Condition vs selling price relationships

This helped in identifying important hidden patterns and relationships within the dataset.

---

# Creation of Encoders and Model Training

We used proper categorical encoding with a **smoothing factor of 10**, which balances the:

* Local mean
* Global mean

to encode the values of a particular categorical feature.

This was followed by:

* Frequency Encoding
* Binary Encoding

for transmission values.

---

# Model Training

We used **CatBoost Regressor** after splitting the validation set as **20 percent** of the dataset.

For hyperparameter tuning, we used:

* `CatBoost`
* `GridSearchCV`

This is because GridSearchCV is faster and more efficient compared to RandomizedSearchCV for our parameter space.

### Model Performance

* **Baseline CatBoost RMSE = 1742.35**
* **Improved CatBoost RMSE = 1603.77**

---

# Deterministic Bit Sequence

Our goal is to create a balance between **aggression and conservatism** during bidding.

We calculate:

[
\text{margin} = \text{predicted_value} - \text{current_highest_bid}
]

If the runway drops to `0`, the function instantly returns `0` (fold).

This ensures that our agent never overpays for an asset.

Next, we calculate:

[
alpha=minimum(0.25,bankroll/1 million)
]

The operator caps alpha at `0.25`, so when our wallet is strong, we are willing to spend up to a maximum scale of **25 percent**.

Since auctions end within a limited time, we use:

[
temporal_velocity=1-e^(-0.25 * auction_round)
]

This helps accelerate bidding behavior as auction rounds progress.

We then compile the dynamic bid increment as a multiple of:

* `margin runway`
* `temporal velocity`

The next bid is placed only if the bid increment remains below a threshold value.

Here, we set the threshold as:

[
100
]

The next bid is calculated as:

[
next_bid=current_highest_bid + maximum(100,bid_increment)
]

Meanwhile, the `auction_round_number` is incremented gradually throughout the process.
