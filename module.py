from sklearn.neighbors import KNeighborsClassifier
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA


iris = datasets.load_iris()
x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)

classify = KNeighborsClassifier(n_neighbors=3, p=2, weights='distance', algorithm='brute')
# n_neighbors: 要取幾個鄰居
# p: 選擇距離的計算方式
# weights: 投票方式為距離等權重或加權
# algorithm:演算法的選擇 (計算效率的考慮)

classify.fit(x_train, y_train)     # training不會有output

result = classify.predict(x_test)
# print(result)

accu = classify.score(x_test, y_test)
print(accu)




# find the best k value ##############################
# accuracy_list = []

# for k in range(1, 100):
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train, y_train)
#     pred = knn.predict(x_test)
#     accuracy_list.append(metrics.accuracy_score(y_test, pred))

# k_range = range(1, 100)
# plt.plot(k_range, accuracy_list)
# plt.show()



# plot ###############################
colors = ['red', 'blue', 'green']
x_decompose = PCA(2).fit_transform(x_test)

plt.figure()
for i in [0, 1, 2]:
    plt.scatter(x_decompose[result==i, 0],
                x_decompose[result==i, 1],
                alpha = 0.7,
                c = colors[i],
                )

plt.show()