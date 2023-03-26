import numpy as np
import matplotlib.pyplot as plt


class K_means:
    def __init__(self, k, mydata):
        self.k = k
        self.means = mydata[[4, 204, 404, 604, 804]]
        self.train(mydata)

    def train(self, mydata):
        is_end = True
        while (is_end):
            classed_points = []
            for i in range(self.k):
                classed_points.append([])
            for point in mydata:
                min_distance = 1e10
                min_index = 6
                for i in range(self.k):
                    dist = np.linalg.norm(point - self.means[i])
                    if dist < min_distance:
                        min_distance = dist
                        min_index = i
                classed_points[min_index].append(point)
            new_means = []
            for i in range(self.k):
                new_means.append([0, 0])
            for i in range(self.k):
                for j in range(len(classed_points[i])):
                    new_means[i] += classed_points[i][j]
                new_means[i] /= len(classed_points[i])
            new_means = np.array(new_means)
            if (self.means == new_means).all():
                is_end = False
            else:
                self.means = new_means


if __name__ == '__main__':
    mean_1 = np.array([1, -1])
    cov_1 = np.array([[1, 0], [0, 1]])
    # 生成高斯分布
    data_1 = np.random.multivariate_normal(mean_1, cov_1, 200)

    mean_2 = np.array([5.5, -4.5])
    cov_2 = np.array([[1, 0], [0, 1]])
    # 生成高斯分布
    data_2 = np.random.multivariate_normal(mean_2, cov_2, 200)

    mean_3 = np.array([1, 4])
    cov_3 = np.array([[1, 0], [0, 1]])
    # 生成高斯分布
    data_3 = np.random.multivariate_normal(mean_3, cov_3, 200)

    mean_4 = np.array([6, 4.5])
    cov_4 = np.array([[1, 0], [0, 1]])
    # 生成高斯分布
    data_4 = np.random.multivariate_normal(mean_4, cov_4, 200)

    mean_5 = np.array([9, 0])
    cov_5 = np.array([[1, 0], [0, 1]])
    # 生成高斯分布
    data_5 = np.random.multivariate_normal(mean_5, cov_5, 200)

    data = np.concatenate((data_1, data_2, data_3, data_4, data_5), axis=0)  # 默认情况下，axis=0可以不写

    model = K_means(5, data)

    print(model.means)

    plt.figure()
    x, y = data.T
    plt.scatter(x, y)
    x1, y1 = model.means.T
    plt.scatter(x1, y1, c='y')
    plt.xlim(-10, 10)
    plt.ylim(-10, 10)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()

Git is a  ++-- distributed version control system.
Git is free software under the GPL.