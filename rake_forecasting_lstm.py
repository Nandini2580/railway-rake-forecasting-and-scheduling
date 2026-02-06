# ================================
# 1. IMPORT LIBRARIES
# ================================
import pandas as pd
import random
import numpy as np
from datetime import datetime, timedelta

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# ================================
# 2. DATA GENERATION
# ================================
N_RECORDS = 1000
START_DATE = datetime(2024, 1, 1)

stations = ["VSKP", "RNC", "BZA", "NAGP", "DURG"]

rake_capacity = {
    "BOXN": 3800,
    "BCN": 2600,
    "BTPN": 3000
}

commodity_map = {
    "BOXN": ["Coal"],
    "BCN": ["Cement", "Foodgrain"],
    "BTPN": ["Fuel"]
}

seasonal_factor_map = {
    "Winter": 0.9,
    "Summer": 1.2,
    "Monsoon": 1.1
}

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    else:
        return "Monsoon"

data = []

for i in range(N_RECORDS):
    date = START_DATE + timedelta(days=i)
    station = random.choice(stations)
    rake_type = random.choice(list(rake_capacity.keys()))
    commodity = random.choice(commodity_map[rake_type])
    capacity = rake_capacity[rake_type]

    actual_load = random.randint(int(capacity * 0.7), capacity)

    season = get_season(date.month)
    seasonal_factor = seasonal_factor_map[season]

    base_demand = random.randint(15, 35)
    available_rakes = random.randint(5, 20)

    demand_pressure_index = round(
        (base_demand * seasonal_factor) / available_rakes, 2
    )

    delay_minutes = int(demand_pressure_index * random.randint(10, 20))

    future_rake_demand = int(base_demand * seasonal_factor + random.randint(-2, 3))

    data.append([
        date.strftime("%Y-%m-%d"),
        station,
        rake_type,
        commodity,
        capacity,
        actual_load,
        delay_minutes,
        demand_pressure_index,
        seasonal_factor,
        future_rake_demand
    ])

columns = [
    "date",
    "station_id",
    "rake_type",
    "commodity_type",
    "capacity_tons",
    "actual_load_tons",
    "delay_minutes",
    "demand_pressure_index",
    "seasonal_factor",
    "future_rake_demand"
]

df = pd.DataFrame(data, columns=columns)


# ================================
# 3. FEATURE ENGINEERING
# ================================
df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek
df["month"] = df["date"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df.drop(columns=["date"], inplace=True)

df_encoded = pd.get_dummies(
    df,
    columns=["station_id", "rake_type", "commodity_type"],
    drop_first=True
)

X = df_encoded.drop(columns=["future_rake_demand"])
y = df_encoded["future_rake_demand"]


# ================================
# 4. SCALING
# ================================
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))


# ================================
# 5. SEQUENCE CREATION
# ================================
def create_sequences(X, y, seq_length=7):
    X_seq, y_seq = [], []
    for i in range(len(X) - seq_length):
        X_seq.append(X[i:i + seq_length])
        y_seq.append(y[i + seq_length])
    return np.array(X_seq), np.array(y_seq)

SEQ_LENGTH = 7
X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LENGTH)


# ================================
# 6. TRAIN TEST SPLIT
# ================================
X_train, X_test, y_train, y_test = train_test_split(
    X_seq,
    y_seq,
    test_size=0.2,
    shuffle=False
)


# ================================
# 7. LSTM MODEL
# ================================
model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(32),
    Dense(1)
])

model.compile(optimizer="adam", loss="mse")
model.summary()


# ================================
# 8. MODEL TRAINING
# ================================
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# ================================
# 9. PREDICTION & INVERSE SCALING
# ================================
y_pred_scaled = model.predict(X_test)

y_test_inv = scaler_y.inverse_transform(y_test)
y_pred_inv = scaler_y.inverse_transform(y_pred_scaled)


# ================================
# 10. ACCURACY METRIC
# ================================
def forecasting_accuracy(y_true, y_pred, epsilon=1e-6):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    return 100 - mape

accuracy = forecasting_accuracy(y_test_inv, y_pred_inv)
print(f"\nLSTM Forecasting Accuracy: {accuracy:.2f}%")
