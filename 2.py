import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


data = pd.read_csv('SSKdataset2.csv')

X = data[['C', 'SetVoltage']]
y = data[['PeakCurrent', 'FWHM']]


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


model = Sequential()
model.add(Dense(64, input_dim=X_scaled.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2))

model.compile(optimizer='adam', loss='mean_squared_error')


model.fit(X_scaled, y, epochs=3     000, batch_size=8, verbose=1)


manual_test_values = [
    {'C': 100, 'SetVoltage': 2100},
    {'C': 100, 'SetVoltage': 2700},
    {'C': 200, 'SetVoltage': 3300},
    {'C': 200, 'SetVoltage': 1800}
]

actual_peakcurrent_values = [720, 1032, 1680, 664]
actual_fwhm_values = [224, 222, 312, 348]

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.float_format', '{:.2f}'.format)


results = []
for i, (test_values, actual_peakcurrent, actual_fwhm) in enumerate(zip(manual_test_values, actual_peakcurrent_values, actual_fwhm_values), start=1):

    manual_input = pd.DataFrame([test_values])
    manual_input_scaled = scaler.transform(manual_input)


    manual_prediction = model.predict(manual_input_scaled)


    predicted_peakcurrent, predicted_fwhm = manual_prediction.flatten()

    result = {
        **test_values,
        'Predicted PeakCurrent': f"{predicted_peakcurrent:.2f}",
        'Actual PeakCurrent': actual_peakcurrent,
        'Predicted FWHM': f"{predicted_fwhm:.2f}",
        'Actual FWHM': actual_fwhm
    }
    results.append(result)

results_df = pd.DataFrame(results)
print("\nResults for Manual Test Inputs:")
print(results_df)

y_pred = model.predict(X_scaled)
y_manual_pred = model.predict(scaler.transform(pd.DataFrame(manual_test_values)))

y_true = np.column_stack((actual_peakcurrent_values, actual_fwhm_values))


mse = mean_squared_error(y_true, y_manual_pred)
r2 = r2_score(y_true, y_manual_pred)

print(f"\nMean Squared Error for Manual Test Set: {mse:.2f}")
print(f"R² Score for Manual Test Set: {r2:.2f}")
