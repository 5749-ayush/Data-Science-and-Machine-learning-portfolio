print("--- Part 1: Importing Necessary Libraries and Dataset ---")

import pandas as pd         # For handling dataframes (tables of data)
import numpy as np          # For numerical operations, especially with arrays
import matplotlib.pyplot as plt # For creating static plots
import seaborn as sns       # For enhanced data visualizations (built on matplotlib)
from sklearn.preprocessing import MinMaxScaler # To scale features (important for neural networks)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # For evaluating model performance
import tensorflow as tf     # The core TensorFlow library for deep learning
from tensorflow.keras.models import Sequential # To create a linear stack of neural network layers
from tensorflow.keras.layers import LSTM, Dense, Dropout # Specific neural network layer types
from tensorflow.keras.callbacks import EarlyStopping # To stop training when model performance plateaus


try:
    get_ipython().run_line_magic('matplotlib', 'inline')
except NameError:
    pass # Ignore if not in an IPython environment

# Loading the dataset
try:
    df = pd.read_csv('MicrosoftStock.csv')
    print("Dataset 'MicrosoftStock.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'MicrosoftStock.csv' not found.")
    print("Please ensure the CSV file is in the same directory as 'predict_stock.py'.")
    exit() # Stop the script if the dataset isn't found

# Explore the dataset's initial structure and content.
print("\n--- Dataset Head (first 5 rows) ---")
print(df.head()) # Shows the top 5 rows to quickly inspect data format

print("\n--- Dataset Information (data types, non-null counts) ---")
df.info() 
print("\n--- Dataset Statistical Description ---")
print(df.describe()) # Generates descriptive statistics (mean, std, min, max, quartiles) for numerical columns.


print(f"\nNumber of unique dates: {df['date'].nunique()}")
print(f"Total number of rows in dataset: {len(df)}")
if df['date'].nunique() != len(df):
    print("Warning: Duplicate dates found. This might require additional review for time-series integrity.")


# --- Part 2: Data Preprocessing ---

print("\n--- Part 2: Data Preprocessing ---")

# Convert the 'date' column to datetime objects
# This will allows for easy time-based operations and plotting.
df['date'] = pd.to_datetime(df['date'])
print("\n'date' column converted to datetime format.")

# Set the 'date' column as the DataFrame's index
# This is standard practice for time-series data, making data slicing by date intuitive.
df.set_index('date', inplace=True)
print("Date column set as DataFrame index.")
print("DataFrame head after setting index:")
print(df.head())

df.drop(columns=['index', 'Name'], inplace=True, errors='ignore')
print("\n'index' and 'Name' columns dropped as they are not needed for forecasting.")
print("DataFrame head after dropping columns:")
print(df.head())

# Handle missing values using interpolation
# For time series data, interpolation is a common method to fill gaps,
# assuming a linear relationship between known values.
numerical_cols = df.select_dtypes(include=np.number).columns # Select only numerical columns for interpolation
if df[numerical_cols].isnull().sum().sum() > 0:
    print("\nMissing values detected in numerical columns. Applying linear interpolation...")
    df[numerical_cols] = df[numerical_cols].interpolate(method='linear', limit_direction='forward', axis=0)
    print("Missing values filled using linear interpolation.")
else:
    print("\nNo missing values found in numerical columns, skipping interpolation.")

print("\nVerifying for any remaining missing values after interpolation:")
print(df.isnull().sum()) # Should show 0 for all numerical columns

# Create additional technical features
# These indicators are derived from historical price and volume data and are
# widely used in financial analysis to help predict future movements.

print("\nCreating additional technical indicators: Moving Averages, Bollinger Bands, RSI...")

# Simple Moving Average (SMA_20): Average closing price over the last 20 days.
# Helps smooth out price data to identify trend direction.
df['SMA_20'] = df['close'].rolling(window=20).mean()

# Exponential Moving Average (EMA_20): Gives more weight to recent prices than older ones.
# More responsive to new information than SMA.
df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

# Bollinger Bands: Volatility indicators, consisting of a middle band (SMA) and two outer bands.
# The outer bands are typically two standard deviations away from the SMA.
df['Middle_Band'] = df['SMA_20'] # The middle band is the SMA
df['Upper_Band'] = df['Middle_Band'] + (df['close'].rolling(window=20).std() * 2) # Upper band
df['Lower_Band'] = df['Middle_Band'] - (df['close'].rolling(window=20).std() * 2) # Lower band

# Relative Strength Index (RSI): A momentum indicator measuring the speed and change of price movements.
# RSI values typically range from 0 to 100.
def calculate_rsi(data, window=14):
    """Calculates the Relative Strength Index (RSI) for a given data series."""
    delta = data.diff() # Calculate the difference in price from the previous day
    gain = delta.where(delta > 0, 0) # Positive price changes (gains)
    loss = (-delta).where(delta < 0, 0) # Negative price changes (losses, converted to positive)

    # Calculate exponential moving average for gains and losses
    avg_gain = gain.ewm(com=window - 1, min_periods=window).mean()
    avg_loss = loss.ewm(com=window - 1, min_periods=window).mean()

    # Calculate Relative Strength (RS)
    rs = avg_gain / avg_loss
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    return rsi

df['RSI'] = calculate_rsi(df['close'], window=14) # Add RSI as a new column

# Drop rows with NaN values resulting from indicator calculations
initial_nan_rows_count = df.isnull().sum(axis=1).iloc[:(max(20, 14))].any() # Check if any NaNs exist in the first few rows
df.dropna(inplace=True)
if initial_nan_rows_count:
    print(f"Dropped initial rows with NaN values (due to rolling window calculations). New dataframe length: {len(df)}")
else:
    print("No additional NaN rows dropped after technical indicator calculations (or dataset was too small).")

print("\nDataFrame head after adding technical indicators and dropping resulting NaNs:")
print(df.head())
print("\nDataFrame tail after adding technical indicators:")
print(df.tail())
print("\nDataFrame info after preprocessing:")
df.info()


# Normalize numerical features using MinMaxScaler
# Scaling data to a range (e.g., 0 to 1) is crucial for neural networks like LSTMs.
# It prevents features with larger values from dominating the learning process.
print("\nNormalizing numerical features using MinMaxScaler...")
features_to_scale = ['open', 'high', 'low', 'close', 'volume',
                     'SMA_20', 'EMA_20', 'Middle_Band', 'Upper_Band', 'Lower_Band', 'RSI']
scaler = MinMaxScaler(feature_range=(0, 1)) # Initialize scaler to scale values between 0 and 1
df_scaled = scaler.fit_transform(df[features_to_scale]) # Fit the scaler to our data and transform it

# Convert the scaled NumPy array back to a Pandas DataFrame
# This preserves column names and index (dates) for easier manipulation.
df_scaled = pd.DataFrame(df_scaled, columns=features_to_scale, index=df.index)
print("Numerical features successfully scaled.")
print("Scaled DataFrame head:")
print(df_scaled.head())


# Split dataset into training (80%) and testing (20%) sets
# For time series data, it's vital to split chronologically.
# Training data comes before testing data to simulate real-world prediction.
train_size = int(len(df_scaled) * 0.8) # Calculate 80% of the data size
train_data = df_scaled.iloc[:train_size] # Select the first 80% for training
test_data = df_scaled.iloc[train_size:]  # Select the remaining 20% for testing

print(f"\nTotal data points: {len(df_scaled)}")
print(f"Training data points (80%): {len(train_data)}")
print(f"Testing data points (20%): {len(test_data)}")

# Create sequences for LSTM
# LSTMs require data in a specific 3D format: (samples, timesteps, features).
# 'timesteps' (or look_back) is the number of past days the model considers for each prediction.
LOOK_BACK = 60 # Define how many previous days' data to use for predicting the next day.

def create_sequences(dataset, look_back=LOOK_BACK):
    """
    Transforms a dataset into sequences suitable for LSTM.
    X contains sequences of 'look_back' days with all features.
    Y contains the 'close' price of the day immediately following the sequence.
    """
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        # X: Sequence of 'look_back' days, including all features.
        X.append(dataset[i:(i + look_back), :])
        # Y: The 'close' price for the day immediately after the sequence (the target).
        # We use .get_loc('close') to find the index of the 'close' column dynamically.
        Y.append(dataset[i + look_back, df_scaled.columns.get_loc('close')])
    return np.array(X), np.array(Y)

# Convert DataFrames to NumPy arrays before creating sequences.
train_array = train_data.values
test_array = test_data.values

X_train, y_train = create_sequences(train_array, LOOK_BACK)
X_test, y_test = create_sequences(test_array, LOOK_BACK)

print(f"\nShape of X_train (samples, timesteps, features): {X_train.shape}")
print(f"Shape of y_train (samples, target_value): {y_train.shape}")
print(f"Shape of X_test (samples, timesteps, features): {X_test.shape}")
print(f"Shape of y_test (samples, target_value): {y_test.shape}")

# Reshape y_test for inverse scaling later. It needs to be 2D (samples, 1).
y_test_reshaped = y_test.reshape(-1, 1)


# --- Part 3: Exploratory Data Analysis (EDA) ---
# This part tell is about  visualizing the data to understand trends, patterns,
# and relationships between different features.

print("\n--- Part 3: Exploratory Data Analysis (EDA) ---")

# Visualize Microsoft stock price trends over time with technical indicators.
# This plot helps to see the overall trend, seasonality (if any), and how
# the calculated indicators track the price.
plt.figure(figsize=(18, 9)) # Set the figure size for better readability
plt.plot(df['close'], label='Historical Close Price', color='blue', alpha=0.8)
plt.plot(df['SMA_20'], label='20-Day Simple Moving Average', color='orange', linestyle='--')
plt.plot(df['EMA_20'], label='20-Day Exponential Moving Average', color='green', linestyle='-.')
plt.plot(df['Upper_Band'], label='Upper Bollinger Band', color='red', linestyle=':')
plt.plot(df['Lower_Band'], label='Lower Bollinger Band', color='red', linestyle=':')
plt.title('Microsoft Stock Price Trend with Technical Indicators')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend() # Display labels for each line
plt.grid(True) # Add a grid for easier reading
plt.tight_layout() # Adjust plot to prevent labels from overlapping
plt.show() # Display the plot

# Visualize Trading Volume over time.
# Volume can indicate the strength of price movements. High volume on a price change
# often signifies a stronger trend.
plt.figure(figsize=(18, 5))
plt.plot(df['volume'], label='Trading Volume', color='purple', alpha=0.7)
plt.title('Microsoft Stock Trading Volume Over Time')
plt.xlabel('Date')
plt.ylabel('Volume')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Analyze price correlations with trading volume and technical indicators.
# A correlation matrix shows how each feature relates to every other feature.
# Values closer to 1 or -1 indicate strong positive or negative correlation, respectively.
print("\nCorrelation Matrix of Features (numerical correlation with each other):")

# We use the original (non-scaled) DataFrame for correlation for interpretability.
correlation_matrix = df[features_to_scale].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Stock Features and Indicators')
plt.tight_layout()
plt.show()

print("\nCorrelation of each feature with the 'close' price (sorted):")
print(correlation_matrix['close'].sort_values(ascending=False))
# This output shows which features are most strongly correlated with the closing price,
# giving insights into their predictive power.

# Identifying seasonal patterns and market trends can also be inferred from the main price trend plot.
# For Microsoft stock, you might observe overall growth trends, but distinct seasonal patterns



# --- Part 4: Model Training and Selection (LSTM using TensorFlow) ---
# This section defines our deep learning model, compiles it, and trains it that will be use for  our prepared training data.

print("\n--- Part 4: Model Training and Selection (LSTM using TensorFlow) ---")

# Define the LSTM model architecture
# A Sequential model is a linear stack of layers.
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2), # Dropout layer: Randomly sets 20% of input units to 0. Helps prevent overfitting.

    # Second LSTM layer:
    # Continues to process the sequence from the previous LSTM layer.
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),

    # Third LSTM layer:
    # `return_sequences=False` (default): The last LSTM layer outputs a single value per sample,
    # as we are predicting a single closing price for the next day.
    LSTM(units=50),
    Dropout(0.2),

    # Dense output layer:
    # `units=1`: Outputs a single numerical value (the predicted closing price).
    Dense(units=1)
])

# Compile the model
# This configures the model for training.
# `optimizer='adam'`: An efficient algorithm for stochastic gradient descent, widely used.
# `loss='mean_squared_error'`: Common loss function for regression problems.
# It calculates the average of the squares of the errors.
model.compile(optimizer='adam', loss='mean_squared_error')
print("\nLSTM model compiled successfully.")

# Setup Early Stopping
# A callback to prevent overfitting. It monitors a metric (validation loss)
# and stops training if it doesn't improve for a specified number of epochs (`patience`).
# `restore_best_weights=True`: Ensures that the model reverts to the weights from the epoch
# that had the best performance on the monitored metric.
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
print("Early stopping callback configured.")

# Train the model
print("\nStarting model training...")
# `model.fit()` trains the neural network.
history = model.fit(
    X_train, y_train,        # Training input (X) and target (y)
    epochs=100,              # Maximum number of times to iterate over the entire dataset
    batch_size=32,           # Number of samples per gradient update
    validation_split=0.1,    # Reserve 10% of the training data for validation during training
    callbacks=[early_stopping], # Apply the early stopping callback
    verbose=1                # Display training progress (1 for progress bar)
)

print("\nModel training complete. Summary of the trained model:")
model.summary() # Prints a summary of the model's layers and parameter counts.

# Plot training & validation loss values
# Visualizing loss helps understand if the model is learning, overfitting, or underfitting.
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Model Loss During Training (Mean Squared Error)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# --- Part 5: Model Evaluation and Prediction ---

print("\n--- Part 5: Model Evaluation and Prediction ---")

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred_scaled = model.predict(X_test) # Predictions are in the scaled format (0 to 1)


# Creating dummy arrays with zeros for all features except the 'close' price
dummy_array_pred = np.zeros((len(y_pred_scaled), len(features_to_scale)))
dummy_array_actual = np.zeros((len(y_test_reshaped), len(features_to_scale)))

# Get the index of the 'close' column in our features_to_scale list
close_col_idx = features_to_scale.index('close')

# Place the scaled predictions and actual values into the 'close' column of the dummy arrays
dummy_array_pred[:, close_col_idx] = y_pred_scaled.flatten() # Flatten converts 2D to 1D array
dummy_array_actual[:, close_col_idx] = y_test_reshaped.flatten()

# Inverse transform only the 'close' price column using the fitted scaler
y_pred = scaler.inverse_transform(dummy_array_pred)[:, close_col_idx]
y_actual = scaler.inverse_transform(dummy_array_actual)[:, close_col_idx]

print("Predictions and actual values inverse transformed to original scale.")

# Evaluate the model using common regression metrics
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print(f"\nModel Evaluation Metrics on Test Set (Original USD Scale):")
print(f"  Mean Absolute Error (MAE): {mae:.4f}")  # Average absolute difference between actual and predicted
print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}") # Standard deviation of the prediction errors
print(f"  R² Score (Coefficient of Determination): {r2:.4f}") # How well the model explains variance (0-1, higher is better)

# Visualize actual vs. predicted prices on the test set
# This plot visually compares the model's predictions against the true prices in the test period.
plt.figure(figsize=(18, 9))
# Plot actual prices from the test set. Note the index adjustment due to LOOK_BACK for X_test.
plt.plot(df.index[train_size + LOOK_BACK:], y_actual, label='Actual Close Price (Test Set)', color='blue')
# Plot predicted prices from the test set.
plt.plot(df.index[train_size + LOOK_BACK:], y_pred, label='Predicted Close Price (Test Set)', color='orange', linestyle='--')
plt.title('Microsoft Stock Price Prediction: Actual vs. Predicted (Test Set)')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Predict Microsoft stock prices for the next 30 days
print("\n--- Predicting Microsoft stock prices for the next 30 days ---")

# Get the last `LOOK_BACK` days from the *scaled* dataset to start the prediction chain.
last_n_days_scaled = df_scaled.iloc[-LOOK_BACK:].values
# Reshape this input to match LSTM's required format: (1, timesteps, features)
current_input = last_n_days_scaled.reshape(1, LOOK_BACK, len(features_to_scale))

future_predictions_scaled = [] # To store scaled future predictions
num_future_days = 30 # Number of days to forecast into the future

# Loop to predict `num_future_days` into the future
for _ in range(num_future_days):
    # 1. Predict the next day's closing price (scaled)
    next_day_pred_scaled = model.predict(current_input)[0][0]

    # 2. Add this prediction to our list
    future_predictions_scaled.append(next_day_pred_scaled)

    # 3. Prepare the input for the next prediction step (rolling window)
   
    new_day_features = np.zeros((len(features_to_scale),))
    new_day_features[close_col_idx] = next_day_pred_scaled # Set the predicted close price

    # For simplicity, copy other features from the last day of the current input sequence.
    # This is a basic auto-regressive approach.
    for i, feature in enumerate(features_to_scale):
        if feature == 'close': # Skip 'close' as it's already set by prediction
            continue
        new_day_features[i] = current_input[0, -1, i] # Copy the last known value for other features

    # Append the new day's features to the current input sequence and remove the oldest day.
    # This effectively slides the prediction window forward.
    current_input = np.append(current_input[:, 1:, :], new_day_features.reshape(1, 1, len(features_to_scale)), axis=1)

# Inverse transform the future predictions to get actual USD values
future_predictions_dummy = np.zeros((num_future_days, len(features_to_scale)))
future_predictions_dummy[:, close_col_idx] = np.array(future_predictions_scaled).flatten()
future_predictions = scaler.inverse_transform(future_predictions_dummy)[:, close_col_idx]

# Generate dates for the future predictions
last_historical_date = df.index[-1] # Get the last date from your historical data
# 'B' frequency means business day. Adjust if you want all calendar days.
future_dates = pd.date_range(start=last_historical_date + pd.Timedelta(days=1),
                             periods=num_future_days, freq='B')

print("\nPredicted 30-day future stock prices (Close) in USD:")
future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close': future_predictions})
future_df.set_index('Date', inplace=True)
print(future_df)


# Final Visualization: Historical data, test set actuals, and future predictions
# This plot provides a comprehensive view of the model's performance and its forecast.
plt.figure(figsize=(20, 10))
plt.plot(df['close'], label='Historical Close Price (Full Dataset)', color='blue', alpha=0.7)
plt.plot(df.index[train_size + LOOK_BACK:], y_actual, label='Actual Close Price (Test Set)', color='green', linewidth=2)
plt.plot(df.index[train_size + LOOK_BACK:], y_pred, label='Predicted Close Price (Test Set)', color='orange', linestyle='--', linewidth=2)
plt.plot(future_df.index, future_predictions, label='Future 30-Day Predictions', color='red', linestyle='-.', linewidth=2)
plt.title('Microsoft Stock Price Prediction: Historical, Test, and Future Forecast')
plt.xlabel('Date')
plt.ylabel('Close Price (USD)')
plt.legend(loc='upper left') # Place legend in a good spot
plt.grid(True)
plt.tight_layout()
plt.show()

print("\n--- Project Execution Complete ---")
print("This script has successfully: ")
print("1. Loaded and preprocessed the Microsoft stock data.")
print("2. Performed Exploratory Data Analysis (EDA) including plotting trends and correlations.")
print("3. Built and trained an LSTM model using TensorFlow.")
print("4. Evaluated the model's performance on a test set.")
print("5. Generated a 30-day forecast for Microsoft's stock price.")

print("\nImportant Note on Future Predictions:")
print("The multi-step forecasting for the next 30 days makes simplifying assumptions (e.g., copying other feature values from the last known data). For highly accurate long-term forecasts, a more sophisticated multi-output model or recursive prediction with more nuanced feature generation would be required.")