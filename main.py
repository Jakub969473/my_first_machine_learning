import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

names = ('Iris-setosa', 'Iris-versicolor', 'Iris-virginica')

data = pd.read_csv(r'C:\Users\torpe\Desktop\iris\iris.data')

data.columns = ['sepal length in cm', 'sepal width in cm', 'petal length in cm',
                'petal width in cm', 'class']

name_to_int = []
for name in data['class']:
    if name == 'Iris-setosa':
        name_to_int.append(1)
    elif name == 'Iris-versicolor':
        name_to_int.append(2)
    else:
        name_to_int.append(3)

flowers = np.array(name_to_int)
idk = []
[idk.append(list(data.iloc[i])[0:-1]) for i in range(0, 149)]
idk = np.array(idk)
clf = LinearRegression()
clf.fit(X=idk, y=flowers)


while True:
    print('Insert data about flower that you want to identify')
    x = list(input().split(','))
    x = [float(i) for i in x]
    pred = clf.predict(np.array([x]))
    print(names[round(pred[0]) - 1])
