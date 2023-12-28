import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

data = pd.read_csv(
    "housing_price_dataset.csv",
    header=0,
    sep=",",)

y = data['Neighborhood']
X = data[['SquareFeet', 'Bedrooms', 'Bathrooms', 'YearBuilt', 'Price']]

scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_encoded, test_size=0.2, random_state=42)

X_train_tensor = torch.Tensor(X_train)
y_train_tensor = torch.LongTensor(y_train)

X_test_tensor = torch.Tensor(X_test)
y_test_tensor = torch.LongTensor(y_test)


train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

class NeuralNetwork(nn.Module):
    def __init__(self,input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x= self.fc2(x)
        return x

input_size = X_train.shape[1]
hidden_size = 64
output_size = len(np.unique(y_train))
model = NeuralNetwork(input_size, hidden_size, output_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        z, predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct/total

print(f'Pytorch Accuracy: {accuracy}')

model = Sequential()
model.add(Dense(units=64, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(units=len(np.unique(y_train)), activation='softmax'))

optimizer = Adam(lr=0.001)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, shuffle=True)
accuracy = model.evaluate(X_test, y_test)[1]
print(f'Keras Accuracy:{accuracy}')