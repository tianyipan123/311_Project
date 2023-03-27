from utils import *

import numpy as np
import matplotlib.pyplot as plt

NUM_ITERATION = 300

def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def neg_log_likelihood(data, theta, beta, a):
    """ Compute the negative log-likelihood.

    :param data: 2D sparse matrix
    :param theta: Vector
    :param beta: Vector
    :param a: Vector
    :return: float
    """
    data = data.toarray()
    exp_mat = create_exp_diff_mat(data.shape, theta, beta)
    a_mat = np.zeros(data.shape) + a.reshape((-1, 1))
    prob_mat = data * np.log(a_mat + exp_mat) + \
               (1 - data) * np.log(1 - a_mat) - np.log(1 + exp_mat)
    log_lklihood = np.nansum(prob_mat)
    return -log_lklihood


def update_parameters(data, lr, lr_a, theta, beta, a):
    """ Update theta, beta, and c using gradient descent.

    :param data: 2D sparse matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :param a: Vector
    :return: tuple of vectors
    """
    data = data.toarray()
    # update theta
    diff_mat = create_diff_mat(data.shape, theta, beta)
    exp_mat = np.exp(diff_mat)
    a_mat = np.zeros(data.shape) + a.reshape((-1, 1))
    theta += lr * np.nansum(data * exp_mat / (a_mat + exp_mat) -
                            sigmoid(diff_mat), axis=1)

    # update beta
    diff_mat = create_diff_mat(data.shape, theta, beta)
    exp_mat = np.exp(diff_mat)
    beta += lr * np.nansum(-data * exp_mat / (a_mat + exp_mat) +
                           sigmoid(diff_mat), axis=0)

    # update a
    exp_mat = create_exp_diff_mat(data.shape, theta, beta)
    a += lr_a * np.nansum(data / (a_mat + exp_mat) -
                        (1 - data) / (1 - a_mat), axis=1)

    return theta, beta, a


def irt(data, val_data, lr, lr_a, iterations):
    """ Train IRT model.

    :param data: 2D sparse matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # parameter initialization
    shape = data.shape
    theta = np.random.randn(shape[0])
    beta = np.random.randn(shape[1])
    a = np.random.randn(shape[0])
    lld_list = []
    val_acc_lst = []

    # training iteration
    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta, a=a)
        score = evaluate(data=val_data, theta=theta, beta=beta, a=a)
        lld_list.append(neg_lld)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta, a = update_parameters(data, lr, lr_a, theta, beta, a)

    return theta, beta, a, lld_list, val_acc_lst


def evaluate(data, theta, beta, a):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :param a: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p = float(a[u])
        p_a = sigmoid(x) * (1 - p) + p
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    # learning
    lr = 0.001
    lr_a = lr / 7

    # train model
    theta, beta, a, lld_list, val_acc_lst = irt(sparse_matrix, val_data, lr, lr_a, NUM_ITERATION)

    # evaluate accuracy
    print(f"train accuracy is {evaluate(train_data, theta, beta, a)}")
    print(f"valid accuracy is {evaluate(val_data, theta, beta, a)}")
    print(f"test accuracy is {evaluate(test_data, theta, beta, a)}")
    return theta, beta, a, lld_list, val_acc_lst


def create_diff_mat(shape, theta, beta):
    """Create the difference matrix of given shape with each entry equal to
    theta[i] - beta[j].

    :param shape: 2-tuple
    :param theta: Vector
    :param beta: Vector
    """
    theta = theta[:, np.newaxis]
    beta = beta[:, np.newaxis]
    inner_mat = np.zeros(shape)
    inner_mat -= beta.reshape(1, -1)
    inner_mat += theta
    return inner_mat


def create_exp_diff_mat(shape, theta, beta):
    """Create the elementwise exponential of difference matrix.

    :param shape: 2-tuple
    :param theta: Vector
    :param beta: Vector
    """
    return np.exp(create_diff_mat(shape, theta, beta))


if __name__ == "__main__":
    theta, beta, a, lld_list, val_acc_lst = main()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    ax1.plot(np.arange(NUM_ITERATION), lld_list, 'b')
    ax2.plot(np.arange(NUM_ITERATION), val_acc_lst, 'r')
    ax1.set_xlabel("#iteraion")
    ax1.set_ylabel("negative log-likelihood")
    ax2.set_ylabel("validation accuracy")
    plt.show()
