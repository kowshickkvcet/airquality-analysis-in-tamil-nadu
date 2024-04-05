import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the CSV file
df = pd.read_csv('feedback.txt')

# Encode 'positive' as 1 and 'negative' as 0
df['sentiment'] = df['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

# Extract features and sentiments
X_tfidf = TfidfVectorizer(max_features=5000).fit_transform(df['text']).toarray()
y = df['sentiment']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(128, input_shape=(X_tfidf.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Use 'sigmoid' for binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Make predictions on the test set
y_pred_proba = model.predict(X_test)

# Apply a threshold (0.5) to get binary predictions
y_pred = (y_pred_proba > 0.5).astype(int)

# Print classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))
