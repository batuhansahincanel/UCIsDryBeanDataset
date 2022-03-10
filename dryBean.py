
import pandas as pd
import numpy as np


dataset = pd.read_excel("Dry_Bean_Dataset.xlsx")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

#Encoding the Depending Variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

#Spreading it as train and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

#Scaling train and test sets
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Building a Random Forest Model
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 0, n_jobs = -1)
classifier.fit(X_train, y_train)

#Prediction 
y_pred = classifier.predict(X_test)

#Accuracy Score & Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
ac = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

#k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
cv_score = cross_val_score(classifier, X_train, y_train, cv = 10, n_jobs=-1)
cv_score_mean = cv_score.mean()
cv_score_std = cv_score.std()


X_test = sc.inverse_transform(X_test)
y_pred = le.inverse_transform(y_pred)