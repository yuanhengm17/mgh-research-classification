import firebase_admin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from firebase_admin import credentials, firestore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

# Set PyTorch to use the CPU as the default device
torch.set_default_device('cpu')

# Firebase Initialization
cred = credentials.Certificate("adminkey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Constants
INCLUDE_ONLY = ['Stephen']
ACTIVITIES = ['walk', 'sit']
CHUNK_SIZE = 2  # in seconds
START_FROM, END_TO = 300, 300  # Trim first/last samples
NUM_CLASSES = 5

# Data Structures
def fetch_data(collection_name, activities, include_only, start_from=300, end_to=300):
    """Fetch and preprocess data from Firestore."""
    data, docs = [], []
    for person in db.collection(collection_name).stream():
        person_name = str(person.to_dict().get('name', ''))
        if person_name not in include_only:
            continue

        for activity in activities:
            for recording in db.collection(collection_name).document(person_name).collection(activity).stream():
                record = recording.to_dict()
                if 'acceleration' not in record:
                    continue

                docs.append(record)
                df = pd.DataFrame(record['acceleration'])[start_from:-end_to]
                data.append(df)
    return data, docs

# Fetch and preprocess training/testing data
training_data_raw, training_docs = fetch_data("training", ACTIVITIES, INCLUDE_ONLY)
testing_data_raw, testing_docs = fetch_data("testing", ACTIVITIES, INCLUDE_ONLY)

# Chunking and Labeling
def chunk_data(data_raw, docs, chunk_size, activities):
    """Split data into chunks and assign labels."""
    data, labels = [], []
    activity_distribution = np.zeros(len(activities))

    for i in range(len(data_raw)):
        num_chunks = len(data_raw[i]) // (chunk_size * 100)
        for j in range(num_chunks):
            x = list(data_raw[i]["x"])[j * chunk_size * 100:(j + 1) * chunk_size * 100]
            y = list(data_raw[i]["y"])[j * chunk_size * 100:(j + 1) * chunk_size * 100]
            z = list(data_raw[i]["z"])[j * chunk_size * 100:(j + 1) * chunk_size * 100]
            activity = docs[i]['activity']
            label = activities.index(activity)

            activity_distribution[label] += 1
            data.append([x, y, z])
            labels.append(label)

    return data, labels, activity_distribution

# Chunk the data
training_data, training_labels, training_distribution = chunk_data(training_data_raw, training_docs, CHUNK_SIZE, ACTIVITIES)
testing_data, testing_labels, testing_distribution = chunk_data(testing_data_raw, testing_docs, CHUNK_SIZE, ACTIVITIES)

# Convert labels to one-hot encoding
def one_hot_encode(labels, num_classes):
    return F.one_hot(torch.tensor(labels, device='cpu').long(), num_classes=num_classes).float()

training_labels = one_hot_encode(training_labels, NUM_CLASSES)
testing_labels = one_hot_encode(testing_labels, NUM_CLASSES)

# Convert data to tensors
training_data = torch.tensor(training_data).float()
testing_data = torch.tensor(testing_data).float()

print(f"Training data: {len(training_data)} segments, distribution: {training_distribution}")
print(f"Testing data: {len(testing_data)} segments, distribution: {testing_distribution}")

# Neural Network Model
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.M1 = nn.Linear(600, 300)
        self.M2 = nn.Linear(300, 100)
        self.M3 = nn.Linear(100, 50)
        self.M4 = nn.Linear(50, NUM_CLASSES)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 600)
        x = self.R(self.M1(x))
        x = self.R(self.M2(x))
        x = self.R(self.M3(x))
        x = self.M4(x)
        return x.squeeze()

# Initialize the model
model = NeuralNet().to("cpu")

# Training Function
def train_model(model, training_data, training_labels, n_epochs=20, learning_rate=0.01):
    optimizer = SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    losses = []
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(len(training_data)):
            x = training_data[i]
            y = training_labels[i]
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(training_data)
        losses.append(avg_loss)
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {avg_loss:.4f}")

    return losses

# Train the model
losses = train_model(model, training_data, training_labels)

# Plot Losses
plt.plot(losses, 'o--')
plt.xlabel("Epoch")
plt.ylabel("Cross Entropy Loss")
plt.title("Training Loss Over Epochs")
plt.show()

# Testing Function
def test_model(model, testing_data, testing_labels):
    correct = 0
    with torch.no_grad():
        for i in range(len(testing_data)):
            x_sample = testing_data[i]
            yhat_sample = model(x_sample)

            if torch.argmax(yhat_sample).item() == torch.argmax(testing_labels[i]).item():
                correct += 1

    accuracy = correct / len(testing_data) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Test the model
test_model(model, testing_data, testing_labels)

# Save the model
torch.jit.script(model).save('saved_model.pt')

# Visualization Function
def visualize_data(docs, data_raw):
    for i in range(len(docs)):
        plt.figure(i)
        plt.plot(data_raw[i]["time"], data_raw[i]["x"], label="X axis")
        plt.plot(data_raw[i]["time"], data_raw[i]["y"], label="Y axis")
        plt.plot(data_raw[i]["time"], data_raw[i]["z"], label="Z axis")
        plt.plot(data_raw[i]["time"], np.sqrt(data_raw[i]["x"]**2 + data_raw[i]["y"]**2 + data_raw[i]["z"]**2), label="Magnitude")
        plt.title(f"({docs[i]['activity']})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Acceleration (g)")
        plt.legend()
    plt.show()

visualize_data(training_docs, training_data_raw)
