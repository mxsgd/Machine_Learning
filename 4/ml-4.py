import numpy as np
import pandas as pd

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

alldata = alldata.dropna()




alldata["Embarked"] = alldata["Embarked"].apply(
    lambda x: 0 if x in ["C"] else 1
)

alldata["Cabin"] = alldata["Cabin"].str.extract('(\d+)', expand=False)
alldata["Ticket"] = alldata["Ticket"].str.extract('(\d+)', expand=False)


alldata = alldata.apply(pd.to_numeric, errors="coerce")
alldata = alldata.dropna()
print(alldata)

