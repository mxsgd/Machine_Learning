import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

alldata = pd.read_csv(
    "titanic.tsv",
    header=0,
    sep="\t",
    usecols=[
        "Survived",
        "PassengerId",
        "Pclass",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
    ],
)
alldata["Sex"] = alldata["Sex"].apply(
    lambda x: 0 if x in ["female"] else 1
)

alldata["Embarked"] = alldata["Embarked"].apply(
    lambda x: 0 if x in ["C"] else 1
)

alldata["Cabin"] = alldata["Cabin"].str.extract('(\d+)', expand=False)
alldata["Ticket"] = alldata["Ticket"].str.extract('(\d+)', expand=False)


alldata = alldata.apply(pd.to_numeric, errors="coerce")
alldata = alldata.dropna()

Features = [
    "Pclass",
    "Sex",
    "Age",
    "Cabin",
]

data = alldata[Features + ["Survived"]]
data_train, data_test = train_test_split(data, test_size=0.2)

y_train = np.ravel(pd.DataFrame(data_train["Survived"]))
x_train = pd.DataFrame(data_train[Features])

model = LogisticRegression()
model.fit(x_train, y_train)

y_expected = np.ravel(pd.DataFrame(data_test["Survived"]))
x_test = pd.DataFrame(data_test[Features])
y_predicted = model.predict(x_test)

print(y_predicted[:10])

error = mean_squared_error(y_expected, y_predicted)

print(f"MSE wynosi {error}")