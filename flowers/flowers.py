#import data
from  sklearn import  datasets

#load data
iris=datasets.load_iris()
x=iris.data
y=iris.target

#create model
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.5)
from sklearn import neighbors
classifier=neighbors.KNeighborsClassifier()

#train model
classifier.fit(x_train,y_train)
predictions=classifier.predict(x_test)

#test model
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions), file=open("output.txt", "a"))