from sklearn.metrics import accuracy_score

def evaluate_model(model, X_train, Y_train, X_test, Y_test):
    X_train_prediction = model.predict(X_train)
    training_data_accuracy = accuracy_score(Y_train, X_train_prediction)
    print('Accuracy score of the training data : ', training_data_accuracy)

    X_test_prediction = model.predict(X_test)
    test_data_accuracy = accuracy_score(Y_test, X_test_prediction)
    print('Accuracy score of the test data : ', test_data_accuracy)

    return X_test_prediction