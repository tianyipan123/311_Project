from utils import *
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from math import ceil

import item_response


def bootstrap(data):
    """Returns a dataset sampled with replacement from the given data.
    """
    return resample(data)


def train_knn(data, k):
    """Randomly samples the data, and returns a trained KNN model with hyperparameter k.
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(bootstrap(data).toarray())
    return mat


def train_irt(data, lr, iterations):
    """Randomly samples the data, and returns a trained IRT model with hyperparemeters
    lr and iterations.
    """
    data = bootstrap(data)
    shape = data.shape
    theta = np.random.randn(shape[0])
    beta = np.random.randn(shape[1])

    for i in range(iterations):
            theta, beta = item_response.update_theta_beta(data, lr, theta, beta)
    return theta, beta


def knn_predict(train_data, valid_data, k):
    """Trains a KNN model with given k on train_data, then
    returns the predictions on valid_data.
    """
    matrix = train_knn(train_data, k)
    return sparse_matrix_predictions(valid_data, matrix)


def IRT_predict(train_data, valid_data, lr, iterations):
    """Trains an IRT model with hyperparameters lr, iterations on train_data, then
    returns predictions on valid_data.

    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    theta, beta = train_irt(train_data, lr, iterations)
    pred = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        pred.append(p_a >= 0.5)

    return np.array(pred)


def majority(*predictions):
    majority = np.array(predictions, dtype=int)
    assert majority.shape[0] == len(predictions)
    half = ceil(len(predictions) / 2)

    # for i in range(0, len(predictions[0])):
    #     count = 0
    #     for pred in predictions:
    #         if pred[i]:
    #                 count += 1
    #
    #     majority.append(count >= half)
    return majority.sum(axis=0) >= half


def main():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Hyperparameters for IRT
    lr = 0.002
    num_iteration = 100

    # Hyperparameters for KNN
    k = 20

    pred1 = knn_predict(sparse_matrix, val_data, k)
    pred2 = IRT_predict(sparse_matrix, val_data, lr, num_iteration)
    pred3 = knn_predict(sparse_matrix, val_data, 25)
    pred = majority(pred1, pred2, pred3)

    knn_accuracy = evaluate(val_data, pred1)
    irt_accuracy = evaluate(val_data, pred2)
    ensemble_accuracy = evaluate(val_data, pred)
    print(f"Ensemble validation accuracy is: {ensemble_accuracy}, KNN validation accuracy is: {knn_accuracy}, IRT validation accuracy is: {irt_accuracy}")


if __name__ == "__main__":
    main()

