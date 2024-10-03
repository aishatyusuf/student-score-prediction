import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

df = pd.read_csv("student scores.csv")

X = df.drop(["Exam_score"], axis=1)
y = df["Exam_score"]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1)

model = LinearRegression()

model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))

model = pickle.load(open("model.pkl", "rb"))