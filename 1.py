
# # import pandas as pd
# # import numpy as np
# # from sklearn.preprocessing import StandardScaler
# # from tensorflow.keras.models import Sequential
# # from tensorflow.keras.layers import Dense
# # import tensorflow as tf
# # from sklearn.metrics import mean_squared_error, r2_score
# # import random
# #
# # random.seed(42)
# # np.random.seed(42)
# # tf.random.set_seed(42)
# # data = pd.read_csv('SSKdataset2.csv')
# # X = data[['Ein', 'C', 'SetVoltage']]
# # y = data['Eout']
# #
# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)
# #
# # model = Sequential()
# # model.add(Dense(64, input_dim=X_scaled.shape[1], activation='relu'))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(1))
# #
# # model.compile(optimizer='adam', loss='mean_squared_error')
# #
# # model.fit(X_scaled, y, epochs=100, batch_size=1, verbose=0)
# #
# # manual_test_values = [
# #     {'Ein': 61, 'C': 100, 'SetVoltage': 2700},
# #     {'Ein': 31, 'C': 200, 'SetVoltage': 3300},
# #     {'Ein': 14.5, 'C': 200, 'SetVoltage': 1800}
# # ]
# #
# # actual_eout_values = [103, 108, 21]
# # results = []
# #
# # for i, (test_values, actual_eout) in enumerate(zip(manual_test_values, actual_eout_values), start=1):
# #
# #     manual_input = pd.DataFrame([test_values])
# #
# #     manual_input_scaled = scaler.transform(manual_input)
# #
# #     manual_prediction = model.predict(manual_input_scaled)
# #
# #     manual_prediction_flat = manual_prediction.flatten()
# #
# #     result = {**test_values, 'Predicted Eout': f"{manual_prediction_flat[0]:.2f}", 'Actual Eout': actual_eout}
# #     results.append(result)
# #
# # results_df = pd.DataFrame(results)
# # pd.options.display.float_format = '{:.2f}'.format
# # print("Results for Manual Test Inputs:")
# # print(results_df)
# # y_pred = model.predict(X_scaled)
# # y_manual_pred = model.predict(scaler.transform(pd.DataFrame(manual_test_values)))
# # mse_manual = mean_squared_error(actual_eout_values, y_manual_pred)
# # r2_manual = r2_score(actual_eout_values, y_manual_pred)
# # print(f"\nR² Score for Manual Test Set: {r2_manual:.2f}")
# # print(f"Mean Squared Error for Manual Test Set: {mse_manual:.2f}
