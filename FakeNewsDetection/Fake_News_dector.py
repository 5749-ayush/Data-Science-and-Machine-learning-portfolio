import pandas as pd
import numpy as np
import re
import string
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# NLTK import and download, wrapped in a try-except for robustness
try:
    import nltk
    from nltk.corpus import stopwords
    # Try downloading stopwords. If it fails due to permissions or network,
    # inform the user or provide a fallback.
    try:
        nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        print("NLTK stopwords not found. Attempting to download...")
        nltk.download('stopwords', quiet=True) # Added quiet=True to avoid verbose output if successful
        print("NLTK stopwords downloaded successfully.")
    _stopwords = set(stopwords.words('english'))
except ImportError:
    print("NLTK library not found. Text cleaning will proceed without stopwords removal.")
    _stopwords = set() # Empty set if NLTK isn't available

# 1. Importing Libraries and Dataset
print("Step 1: Importing Libraries and Dataset")

# Load the dataset
# Ensure 'news.csv' is in the same directory as this script.
try:
    df = pd.read_csv('news.csv')
    print("Dataset 'news.csv' loaded successfully.")
except FileNotFoundError:
    print("Error: 'news.csv' not found. Please ensure the dataset file is in the correct directory.")
    exit() # Exit if the file isn't found

# Display basic information about the dataset
print("\nDataset Head:")
print(df.head())
print("\nDataset Info:")
df.info()
print("\nDataset Shape:", df.shape)
print("\nLabel Distribution:")
print(df['label'].value_counts())

# Keep only 'text' and 'label' columns as per project objective
# This assumes 'text' and 'label' are consistently named in the CSV.
required_columns = ['text', 'label']
if not all(col in df.columns for col in required_columns):
    print(f"Error: Required columns {required_columns} not found in the dataset.")
    print(f"Available columns: {df.columns.tolist()}")
    exit()
df = df[required_columns]
print("\nDataFrame after selecting relevant columns:")
print(df.head())

# 2. Preprocessing Dataset
print("\nStep 2: Preprocessing Dataset")

# Handle potential missing values
initial_rows = df.shape[0]
df.dropna(inplace=True)
if df.shape[0] < initial_rows:
    print(f"Removed {initial_rows - df.shape[0]} rows with missing values.")
else:
    print("No missing values found in the dataset.")

# Convert labels (FAKE, REAL) into binary values (0,1)
# Handle potential non-standard labels by re-mapping only expected values
label_mapping = {'FAKE': 0, 'REAL': 1}
# Only map values that exist in label_mapping, coerce others to NaN
df['label'] = df['label'].map(label_mapping)

# Check for any labels that weren't mapped (e.g., unexpected values in 'label' column)
unmapped_labels_count = df['label'].isnull().sum()
if unmapped_labels_count > 0:
    print(f"Warning: {unmapped_labels_count} rows had unmappable 'label' values and will be dropped.")
    df.dropna(subset=['label'], inplace=True) # Drop rows where label couldn't be mapped
    df['label'] = df['label'].astype(int) # Ensure type is integer after dropping NaNs
else:
    df['label'] = df['label'].astype(int) # Ensure type is integer if no NaNs

print("\nLabel mapping (FAKE: 0, REAL: 1):")
print(df['label'].value_counts())

# Function to remove stopwords, punctuation, and special characters
def clean_text(text):
    if not isinstance(text, str):
        return "" # Ensure input is a string
    text = text.lower() # Convert to lowercase
    text = re.sub(r'\[.*?\]', '', text) # Remove text in square brackets (e.g., citation numbers)
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Remove URLs
    text = re.sub(r'<.*?>+', '', text) # Remove HTML tags (if any)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text) # Remove punctuation
    text = re.sub(r'\n', '', text) # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text) # Remove words containing numbers
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces and strip leading/trailing spaces
    
    # Remove stopwords using the global _stopwords set
    if _stopwords: # Only attempt if stopwords were loaded successfully
        text = ' '.join(word for word in text.split() if word not in _stopwords)
    # else: # This else block is no longer needed as the message is printed once at the start.
        # print("Note: Stopwords removal skipped as NLTK or stopwords corpus was not available.")
    return text

df['cleaned_text'] = df['text'].apply(clean_text)
print("\nText after cleaning (first 5 entries):")
print(df['cleaned_text'].head())


# Split dataset into training and testing sets (80%-20%)
# Ensure the dataframes are not empty after cleaning and mapping
if df.empty:
    print("Error: DataFrame is empty after preprocessing. Cannot proceed with splitting.")
    exit()

X = df['cleaned_text']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 

print(f"\nTraining set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# Convert text data into numerical format using Tokenization
# vocab_size: the maximum number of words to keep, based on word frequency.
# oov_token: a token for out-of-vocabulary words.
# For future large datasets, consider increasing vocab_size based on corpus size.
vocab_size = 20000 
embedding_dim = 128 # Dimension of the word embeddings

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences of integers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Determine maximum sequence length for padding dynamically but with a cap
all_text_lengths = [len(s) for s in tokenizer.texts_to_sequences(df['cleaned_text'])]
# If the dataset is large, analyze distribution: pd.Series(all_text_lengths).quantile(0.95)
# For the given small snippet, a fixed number or max_len is reasonable.
max_sequence_length_in_data = max(all_text_lengths) if all_text_lengths else 0
# IMPORTANT: Adjusted maxlen and default, consider reducing this if OOM persists
# A good starting point for maxlen can be the 90th or 95th percentile of your article lengths.
# If articles are extremely long, truncation might be necessary.
maxlen = min(max_sequence_length_in_data + 10, 256) # Reduced from 512 to 256 as a first step to save memory

# Ensure maxlen is at least 1 for empty sequences.
if maxlen == 0:
    maxlen = 128 # Fallback if no text data or very short texts

print(f"\nCalculated maximum sequence length for padding: {maxlen}")


X_train_padded = pad_sequences(X_train_sequences, maxlen=maxlen, padding='post', truncating='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=maxlen, padding='post', truncating='post')

print("\nText to sequences and padding complete. Shape of padded sequences:")
print(f"Training padded shape: {X_train_padded.shape}")
print(f"Testing padded shape: {X_test_padded.shape}")

# 3. Generating Word Embeddings (Implemented as part of Model Architecture)
print("\nStep 3: Generating Word Embeddings (will be part of the Keras Embedding Layer)")
print(f"An Embedding layer with output dimension {embedding_dim} will be initialized randomly and learned during model training.")
print("This approach is suitable for text classification and scales well with larger datasets, as the embeddings are learned directly from your text data.")

# 4. Model Architecture
print("\nStep 4: Model Architecture")

# Define a Sequential Deep Learning Model using TensorFlow
# Input Layer (Embedding Layer): Maps word indices to dense vectors.
# LSTM (Long Short-Term Memory) Layer: Captures long-range dependencies in text.
# Dense Layers for classification: Standard fully-connected layers.
# Activation function: Sigmoid for binary classification (output between 0 and 1).

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen, name='embedding_layer'),
    LSTM(128, name='lstm_layer'), # Output 128 LSTM units
    Dense(64, activation='relu', name='dense_1'), # First Dense layer with ReLU
    Dropout(0.5, name='dropout_1'), # Dropout for regularization
    Dense(32, activation='relu', name='dense_2'), # Second Dense layer with ReLU (added for complexity)
    Dropout(0.3, name='dropout_2'), # Another dropout layer
    Dense(1, activation='sigmoid', name='output_layer') # Output layer with Sigmoid for binary classification
])

# Compile the model
# Using Binary Crossentropy Loss for binary classification.
# Adam Optimizer for efficient gradient descent.
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

print("\nModel Summary:")
model.summary()

# 5. Model Training and Prediction
print("\nStep 5: Model Training and Prediction")

# Train the model on training data
# epochs: Number of times to iterate over the entire dataset. For larger datasets, fewer epochs might suffice.
# batch_size: Number of samples per gradient update.
# validation_split: Fraction of the training data to be used as validation data.
print("\nInitiating model training...")
history = model.fit(
    X_train_padded,
    y_train,
    epochs=10, # Can be increased if model is still learning or reduced if overfitting
    # --- REDUCED BATCH SIZE TO SAVE GPU MEMORY ---
    batch_size=16, 
    validation_split=0.1,
    verbose=1 # Show progress bar during training
)

print("\nModel Training Complete.")

# Evaluate model performance using the test set
print("\nEvaluating Model Performance on Test Set:")
loss, accuracy = model.evaluate(X_test_padded, y_test, verbose=0)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# Make predictions on the test set to calculate precision, recall, and F1-score
y_pred_probs = model.predict(X_test_padded)
y_pred = (y_pred_probs > 0.5).astype(int) # Convert probabilities to binary class predictions (0 or 1)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-score: {f1:.4f}")

print("\nThese metrics indicate how well the model performed:")
print("- **Accuracy:** Overall correctness of predictions.")
print("- **Precision:** Proportion of positive identifications that were actually correct.")
print("- **Recall:** Proportion of actual positives that were identified correctly.")
print("- **F1-score:** Harmonic mean of precision and recall, balancing both.")

# Test the model on new unseen news articles
print("\nTesting the trained model on new, unseen news articles:")

# Example fake news article (crafted to sound conspiratorial or sensational)
new_fake_article_1 = "BREAKING! Secret Alien Base Found In Moon's Tycho Crater! NASA hid evidence for decades, new documents reveal. This changes everything you thought you knew about space exploration and our place in the universe. More details to follow on this shocking discovery."
# Example real news article (crafted to sound factual and news-like)
new_real_article_1 = "U.S. Secretary of State John F. Kerry said Monday that he will stop in Paris later this week, amid criticism that no top American officials attended Sunday’s unity march against terrorism. He plans to meet with French officials to discuss international cooperation on security matters."

new_fake_article_2 = "SHOCKING! Michele Obama & Hillary Caught Glamorizing Date Rape Promoters! Infowars.com reports on complete hypocrisy as White House hosts and promotes rappers who boast about assaulting women. Mainstream media silent!"
new_real_article_2 = "U.S. economy added 271,000 jobs in October, beating expectations. The Labor Department reported robust growth, with the unemployment rate falling to 5 percent. Wage growth also saw a significant rise, signaling economic strength."

test_articles = {
    "Fake News Example 1": new_fake_article_1,
    "Real News Example 1": new_real_article_1,
    "Fake News Example 2": new_fake_article_2,
    "Real News Example 2": new_real_article_2,
}

for name, article_text in test_articles.items():
    cleaned_article = clean_text(article_text)
    article_sequence = tokenizer.texts_to_sequences([cleaned_article])
    article_padded = pad_sequences(article_sequence, maxlen=maxlen, padding='post', truncating='post')
    
    prediction_prob = model.predict(article_padded)[0][0]
    predicted_class = 'REAL' if prediction_prob > 0.5 else 'FAKE'
    
    print(f"\n--- {name} ---")
    print(f"Original: '{article_text[:100]}...'")
    print(f"Cleaned: '{cleaned_article[:100]}...'")
    print(f"Prediction Probability (of REAL): {prediction_prob:.4f}")
    print(f"Predicted Class: {predicted_class}")

print("\n--- Project Conclusion ---")
print("This deep learning model, built with TensorFlow, demonstrates the capability to classify news articles as FAKE or REAL based on their text content. The steps covered include data loading, thorough text preprocessing, tokenization, sequence padding, defining a Sequential model with Embedding, LSTM, and Dense layers, and finally, training and evaluating the model.")
