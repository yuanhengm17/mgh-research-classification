import firebase_admin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from firebase_admin import credentials, firestore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

# Set device to CPU
torch.set_default_device('cpu')

# Initialize Firebase
cred = credentials.Certificate("adminkey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Initialize lists to store data
df = []
testDf = []

doc = []
testDoc = []

# Specify the people to include
includeOnly = ['Stephen', 'Ren', 'Ethan Shao', 'Jerry', 'Lillian', 'Michael He', 'Sophia', 'Xin', 'Yuanheng']  # Update as needed

# Update the activities list with all possible activities
activities = ['walk', 'run', 'jump', 'sit', 'upstair']  # Define your activities here
num_classes = len(activities)  # Number of output classes matches the number of activities


# Define chunk size
chunk_size = 2  # in seconds
samples_per_second = 100  # 100Hz
samples_per_chunk = chunk_size * samples_per_second

# Trim parameters (first and last 3 seconds)
startFrom, endTo = 300, 300  # 300 samples = 3 seconds

# Load training data
for person_doc in db.collection("training").stream():
    person = person_doc.to_dict().get('name', '')
    if person not in includeOnly:
        continue

    for activity in activities:
        activity_collection = db.collection("training").document(person).collection(activity)
        for recording in activity_collection.stream():
            recording_dict = recording.to_dict()
            doc.append(recording_dict)
            acceleration = recording_dict.get('acceleration', {})
            df.append(pd.DataFrame(acceleration))

# Load testing data
for person_doc in db.collection("testing").stream():
    person = person_doc.to_dict().get('name', '')
    if person not in includeOnly:
        continue

    for activity in activities:
        activity_collection = db.collection("testing").document(person).collection(activity)
        for recording in activity_collection.stream():
            recording_dict = recording.to_dict()
            testDoc.append(recording_dict)
            acceleration = recording_dict.get('acceleration', {})
            testDf.append(pd.DataFrame(acceleration))

# Trim the data
for i in range(len(df)):
    df[i] = df[i][startFrom:-endTo]

for i in range(len(testDf)):
    testDf[i] = testDf[i][startFrom:-endTo]

# Initialize data and labels
training_data = []
training_label = []
testing_data = []
testing_label = []

# Initialize distribution counters
training_activities_chunk_distribution = np.zeros(len(activities))
testing_activities_chunk_distribution = np.zeros(len(activities))

# Process training data
for i in range(len(df)):
    num_chunks = len(df[i]) // samples_per_chunk
    for j in range(num_chunks):
        x = df[i]["x"].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        y = df[i]['y'].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        z = df[i]['z'].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        activity = doc[i]['activity']
        if activity not in activities:
            continue  # Skip unknown activities
        activity_idx = activities.index(activity)

        training_activities_chunk_distribution[activity_idx] += 1
        training_data.append([x, y, z])
        training_label.append(activity_idx)

# Process testing data
for i in range(len(testDf)):
    num_chunks = len(testDf[i]) // samples_per_chunk
    for j in range(num_chunks):
        x = testDf[i]["x"].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        y = testDf[i]['y'].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        z = testDf[i]['z'].iloc[j*samples_per_chunk : (j+1)*samples_per_chunk].tolist()
        activity = testDoc[i]['activity']
        if activity not in activities:
            continue  # Skip unknown activities
        activity_idx = activities.index(activity)

        testing_activities_chunk_distribution[activity_idx] += 1
        testing_data.append([x, y, z])
        testing_label.append(activity_idx)

# Display dataset information
training_size = len(training_data)
testing_size = len(testing_data)
print(f"Total of {training_size} segments of {chunk_size}-second data available for training: {training_activities_chunk_distribution}")
print(f"Total of {testing_size} segments of {chunk_size}-second data available for testing: {testing_activities_chunk_distribution}")

# One-hot encode labels
training_label = F.one_hot(torch.tensor(training_label).long(), num_classes=num_classes).float()
testing_label = F.one_hot(torch.tensor(testing_label).long(), num_classes=num_classes).float()

# Convert data to tensors
training_data = torch.tensor(training_data).float()
testing_data = torch.tensor(testing_data).float()
for i in range(len(training_data)):

    print(training_label[i])
    print(training_data[i][0].numpy())
    print(training_data[i][1].numpy())
    print(training_data[i][2].numpy())
    print("/////")

class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.M1 = nn.Linear(600, 300)
        self.M2 = nn.Linear(300, 100)
        self.M3 = nn.Linear(100, 50)
        self.M4 = nn.Linear(50, 5)
        self.R = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 600)
        x = self.R(self.M1(x))
        x = self.R(self.M2(x))
        x = self.R(self.M3(x))
        x = self.M4(x)
        return x.squeeze()


f = NeuralNet().to("cpu")


def train_model(n_epochs=20):
    opt = SGD(f.parameters(), lr=0.01)
    L = nn.CrossEntropyLoss()

    losses = []
    epochs = []
    for epoch in range(n_epochs):
        print(f"Epoch {epoch}")
        for i in range(training_size):
            x = training_data[i]
            y = training_label[i]
            opt.zero_grad()
            loss_value = L(f(x), y)
            loss_value.backward()
            opt.step()

            epochs.append(epoch+i/training_size)
            losses.append(loss_value.item())
    return np.array(epochs), np.array(losses)


epoch_data, loss_data = train_model()
epoch_data_avgd, loss_data_avgd = epoch_data.reshape(20, -1).mean(axis=1), loss_data.reshape(20, -1).mean(axis=1)
plt.plot(epoch_data, loss_data, 'o--')
plt.xlabel("Epoch number")
plt.ylabel("Cross entropy")
plt.title("Cross entropy per batch")
plt.show()


def test_model():
    correct = 0

    for i in range(testing_size):
        x_sample = testing_data[i]
        yhat_sample = f(x_sample)

        print(f"Answer: {testing_label[i]}, Predicted: {yhat_sample}")
        if torch.argmax(yhat_sample).item() == torch.argmax(testing_label[i]).item():
            correct += 1
            print("CORRECT")
        else:
            print("WRONG")

    print(f"Test result: {correct}/{testing_size} correct, {correct/testing_size*100}% accuracy")


test_model()

model = torch.jit.script(f)
model.save('saved_model.pt')


def showGraph():
    for i in range(len(doc)):
        plt.figure(i)
        plt.plot(df[i]["time"], df[i]["x"], label="X axis")
        plt.plot(df[i]["time"], df[i]["y"], label="Y axis")
        plt.plot(df[i]["time"], df[i]["z"], label="Z axis")
        plt.plot(df[i]["time"], np.sqrt(df[i]["x"]**2+df[i]["y"]**2+df[i]['z']**2), label="Magnitude")
        plt.title(f"({doc[i]['activity']})")
        plt.xlabel("Time (ms)")
        plt.ylabel("Acceleration (g)")
        plt.legend()
    plt.show()
showGraph()
