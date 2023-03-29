from utils import *
from sklearn.utils import resample
from sklearn.impute import KNNImputer
from math import ceil

import item_response
import neural_network
import torch


def bag(data, num_bags):
    """Returns num_bags datasets sampled with replacement from the given data.
    """
    bags = []
    for i in range(num_bags):
        bags.append(resample(data, replace=True))
    return bags

def train_knn(data, k):
    """Randomly samples the data, and returns a trained KNN model with hyperparameter k.
    """
    nbrs = KNNImputer(n_neighbors=k)
    mat = nbrs.fit_transform(data.toarray())
    return mat


def train_irt(data, lr, iterations):
    """Randomly samples the data, and returns a trained IRT model with hyperparemeters
    lr and iterations.
    """
    shape = data.shape
    theta = np.random.randn(shape[0])
    beta = np.random.randn(shape[1])

    for i in range(iterations):
            theta, beta = item_response.update_theta_beta(data, lr, theta, beta)
    return theta, beta


def knn_predict(train_data, valid_data, test_data, k):
    """Trains a KNN model with given k on train_data, then
    returns the predictions on valid_data.
    """
    matrix = train_knn(train_data, k)
    return sparse_matrix_predictions(valid_data, matrix), sparse_matrix_predictions(test_data, matrix)


def IRT_predict(train_data, valid_data, test_data, lr, iterations):
    """Trains an IRT model with hyperparameters lr, iterations on train_data, then
    returns predictions on valid_data.

    :param valid_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    """
    theta, beta = train_irt(train_data, lr, iterations)
    pred = []
    t_pred = []
    for i, q in enumerate(valid_data["question_id"]):
        u = valid_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        pred.append(p_a >= 0.5)

    for i, q in enumerate(test_data["question_id"]):
        u = test_data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = item_response.sigmoid(x)
        t_pred.append(p_a >= 0.5)

    return np.array(pred), np.array(t_pred)

def train_and_predict_nn(train_data, valid_data, test_data, k, lr, lamb, num_epoch):
    zero_train_matrix = train_data.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_data)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_data = torch.FloatTensor(train_data)
    model = neural_network.AutoEncoder(train_data.shape[1], k)

    neural_network.train(model, lr, lamb, train_data, zero_train_matrix,
              valid_data, num_epoch, False, True)
    
    predictions = []
    test_predictions = []
    for i, u in enumerate(valid_data["user_id"]):
        inputs = torch.autograd.Variable(zero_train_matrix[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        predictions.append(guess)

    for i, u in enumerate(test_data["user_id"]):
        inputs = torch.autograd.Variable(zero_train_matrix[u]).unsqueeze(0)
        output = model(inputs)
        guess = output[0][test_data["question_id"][i]].item() >= 0.5
        test_predictions.append(guess)

    return predictions, test_predictions

def majority(predictions):
    """Returns the majority vote of the given predidctions."""
    majority = np.array(predictions, dtype=int)
    assert majority.shape[0] == len(predictions)
    half = ceil(len(predictions) / 2)
    return majority.sum(axis=0) >= half


def main():
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # Hyperparameters for IRT
    lr = 0.002
    num_iteration = 50

    # Hyperparameters for KNN
    k = 100

    # Hyperparameters for NN
    hl = 250
    nnlr = 0.002
    num_epoch = 30

    # Number of bags (should be divisible by 3)
    N = 9

    datasets = bag(sparse_matrix, N)
    val_predictions = []
    t_predictions = []
    for i in range(N // 3):
        val_preds, test_preds = IRT_predict(datasets[i], val_data, test_data, lr, num_iteration)
        val_predictions.append(val_preds)
        t_predictions.append(test_preds)

    for i in range(N // 3, 2 * N // 3): 
        val_preds, test_preds = knn_predict(datasets[i], val_data, test_data, k)
        val_predictions.append(val_preds)
        t_predictions.append(test_preds)

    for i in range(2 * N // 3, N):
        val_preds, test_preds = train_and_predict_nn(
            datasets[i].toarray(),
            val_data, 
            test_data,
            hl,
            nnlr,
            0.01,
            num_epoch
        )
        val_predictions.append(val_preds)
        t_predictions.append(test_preds)

    val_averaged_prediction = majority(val_predictions)
    t_averaged_prediction = majority(t_predictions)

    print(f"KNN validation accuracy is: {evaluate(val_data, val_predictions[0])}")
    print(f"IRT validation accuracy is: {evaluate(val_data, val_predictions[N // 3])}")
    print(f"NN validation accuracy is: {evaluate(val_data, val_predictions[N-1])}")
    print(f"Ensembled validation accuracy is: {evaluate(val_data, val_averaged_prediction)}")

    print(f"KNN test accuracy is: {evaluate(test_data, t_predictions[0])}")
    print(f"IRT test accuracy is: {evaluate(test_data, t_predictions[N // 3])}")
    print(f"NN test accuracy is: {evaluate(test_data, t_predictions[N-1])}")
    print(f"Ensembled test accuracy is: {evaluate(test_data, t_averaged_prediction)}")

if __name__ == "__main__":
    main()

