import firebase_admin
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from firebase_admin import credentials
from firebase_admin import firestore
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

torch.set_default_device('cuda')


cred = credentials.Certificate("adminKey.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

df = []
testDf = []

doc = []
testDoc = []

includeOnly = ['baoren']

activities = ['walk', 'run', 'jump']
for person in db.collection("training").stream():
    person = str(person.to_dict()['name'])
    if person not in includeOnly:
        continue

    for activity in activities:
        for recording in db.collection("training").document(person).collection(activity).stream():
            doc.append(recording.to_dict())
            df.append(pd.DataFrame(recording.to_dict()['acceleration']))

for person in db.collection("testing").stream():
    person = str(person.to_dict()['name'])
    if person not in includeOnly:
        continue

    for activity in activities:
        for recording in db.collection("testing").document(person).collection(activity).stream():
            testDoc.append(recording.to_dict())
            testDf.append(pd.DataFrame(recording.to_dict()['acceleration']))


startFrom, endTo = 300, 300
# take data from first 3 seconds to last 3 seconds

for i in range(len(df)):
    df[i] = df[i][startFrom:-endTo]

for i in range(len(testDf)):
    testDf[i] = testDf[i][startFrom:-endTo]

training_data = []
training_label = []
testing_data = []
testing_label = []
chunk_size = 2

training_activities_chunk_distribution = np.zeros(5)
testing_activities_chunk_distribution = np.zeros(5)

for i in range(len(df)):
    # split each document into many 2-second chunks
    for j in range(len(df[i]) // (chunk_size*100)):
        x = list(df[i]["x"])[j*chunk_size*100:(j+1)*chunk_size*100]
        y = list(df[i]['y'])[j*chunk_size*100:(j+1)*chunk_size*100]
        z = list(df[i]['z'])[j*chunk_size*100:(j+1)*chunk_size*100]
        activity = doc[i]['activity']
        activity = activities.index(activity)

        training_activities_chunk_distribution[activity] += 1
        training_data.append([x, y, z])
        training_label.append(activity)


for i in range(len(testDf)):
    for j in range(len(testDf[i]) // (chunk_size*100)):
        x = list(testDf[i]["x"])[j*chunk_size*100:(j+1)*chunk_size*100]
        y = list(testDf[i]['y'])[j*chunk_size*100:(j+1)*chunk_size*100]
        z = list(testDf[i]['z'])[j*chunk_size*100:(j+1)*chunk_size*100]
        activity = testDoc[i]['activity']
        activity = activities.index(activity)

        testing_activities_chunk_distribution[activity] += 1
        testing_data.append([x, y, z])
        testing_label.append(activity)

training_size = len(training_data)
testing_size = len(testing_data)
print(f"Total of {training_size} segments of 2-second data available for training: {training_activities_chunk_distribution}")
print(f"Total of {testing_size} segments of 2-second data available for testing: {testing_activities_chunk_distribution}")

training_label = F.one_hot(torch.tensor(training_label, device='cpu').long(), num_classes=5).numpy()
testing_label = F.one_hot(torch.tensor(testing_label, device='cpu').long(), num_classes=5).numpy()

training_label = torch.tensor(training_label).float()
training_data = torch.tensor(training_data)
testing_label = torch.tensor(testing_label).float()
testing_data = torch.tensor(testing_data)


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


f = NeuralNet().to("cuda")


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


# def showGraph():
#     for i in range(len(doc)):
#         plt.figure(i)
#         plt.plot(df[i]["time"], df[i]["x"], label="X axis")
#         plt.plot(df[i]["time"], df[i]["y"], label="Y axis")
#         plt.plot(df[i]["time"], df[i]["z"], label="Z axis")
#         plt.plot(df[i]["time"], np.sqrt(df[i]["x"]**2+df[i]["y"]**2+df[i]['z']**2), label="Magnitude")
#         plt.title(f"{docNames[i]} ({doc[i]['activity']})")
#         plt.xlabel("Time (ms)")
#         plt.ylabel("Acceleration (g)")
#         plt.legend()
#     plt.show()
#
# showGraph()
