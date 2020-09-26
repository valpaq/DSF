from sklearn import linear_model


class Estimator:
    @staticmethod
    def fit(train_x, train_y):
        return linear_model.ElasticNetCV(alphas=[0.0001, 0.0005, 0.001, 0.01, 0.1, 1, 10],
             l1_ratio=[.01, .1, .5, .9, .99], max_iter=5000).fit(x_train, y_train)


    @staticmethod
    def predict(trained, test_x):
        return trained.predict(test_x)
