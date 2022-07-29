from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# load data set
iris = datasets.load_iris(as_frame=True)
iris_data = iris.data
iris_answer = iris.target
iris_dict = iris_data.to_dict()

# 打散data 並分成train跟test兩個group 
data_train, data_test, answer_train, answer_test = train_test_split(iris_data, iris_answer, test_size=0.3, random_state=1)



def distance(a: int, b: int):
    sepal_l = (iris_dict['sepal length (cm)'][a] - iris_dict['sepal length (cm)'][b]) ** 2
    sepal_w = (iris_dict['sepal width (cm)'][a] - iris_dict['sepal width (cm)'][b]) ** 2
    petal_l = (iris_dict['petal length (cm)'][a] - iris_dict['petal length (cm)'][b]) ** 2
    petal_w = (iris_dict['petal width (cm)'][a] - iris_dict['petal width (cm)'][b]) ** 2
    return (sepal_l + sepal_w + petal_l + petal_w) ** 0.5


def collect_index(dataframe):
    collected = []
    for index, row in dataframe.iterrows():
        collected.append(index)
    
    return collected


def determine_class(list):
    a = list.count(0)
    b = list.count(1)
    c = list.count(2)
    if max(a, b, c) == a:
        return 0
    elif max(a, b, c) == b:
        return 1
    else:
        return 2
    
    
def classification_knn(data_train, answer_train, data_test, k):
    training_index = collect_index(data_train)
    testing_index = collect_index(data_test)
    predicted_result = pd.Series(0, index=testing_index)
    for num in testing_index:
        temp_series = pd.Series(0, index=training_index)
        for train_num in training_index:
            dist = distance(train_num, num)
            temp_series[train_num] = dist
        k_list = [index for index, value in temp_series.nsmallest(k).items()]
        k_class = [value for index, value in answer_train.items() if index in k_list]
        predicted_result[num] = determine_class(k_class)

    return predicted_result

my_result = classification_knn(data_train, answer_train, data_test, 5)

# plot ###############################
colors = ['red', 'blue', 'green']
x_decompose = PCA(2).fit_transform(data_test)

plt.figure()
for i in [0, 1, 2]:
    plt.scatter(x_decompose[my_result==i, 0],
                x_decompose[my_result==i, 1],
                alpha = 0.7,
                c = colors[i],
                )

plt.show()

def accuracy_calculate():
    score = 0
    for index, value in my_result.items():
        if value == int(answer_test[index]):
            score += 1
    correct_ratio = round(score/len(my_result)*100, 2)  
    print(f"accuracy ratio = {correct_ratio}%")

accuracy_calculate()


    





