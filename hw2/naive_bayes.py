import data


test_X, test_y = data.load_test()
print(test_X[0])

train_x, train_y = data.load_train()


data.visualize(train_x[0])
