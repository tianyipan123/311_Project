from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def neg_log_likelihood(data, theta, beta):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: 2D sparse matrix
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    diff_mat = create_diff_mat(data.shape, theta, beta)
    prob_mat = data.toarray() * diff_mat - np.log(1 + np.exp(diff_mat))
    log_lklihood = np.nansum(prob_mat)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return -log_lklihood


def update_theta_beta(data, lr, theta, beta):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: 2D sparse matrix
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    #####################################################################
    # TODO:                                                             #
    # Implement the function as described in the docstring.             #
    #####################################################################
    data = data.toarray()
    # update theta
    diff_mat = create_diff_mat(data.shape, theta, beta)
    theta += lr * np.nansum(data - sigmoid(diff_mat), axis=1)

    # update beta
    diff_mat = create_diff_mat(data.shape, theta, beta)
    beta += lr * np.nansum(-data + sigmoid(diff_mat), axis=0)
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
    return theta, beta


def irt(data, val_data, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: 2D sparse matrix
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    # TODO: Initialize theta and beta.
    shape = data.shape
    theta = np.random.randn(shape[0])
    beta = np.random.randn(shape[1])

    val_acc_lst = []
    lld_list = []
    val_lld_lst = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(data, theta=theta, beta=beta)
        lld_list.append(neg_lld)
        score = evaluate(data=val_data, theta=theta, beta=beta)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {}".format(neg_lld, score))
        theta, beta = update_theta_beta(data, lr, theta, beta)

    # TODO: You may change the return values to achieve what you want.
    return theta, beta, val_acc_lst


def evaluate(data, theta, beta):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (theta[u] - beta[q]).sum()
        p_a = sigmoid(x)
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    # You may optionally use the sparse matrix.
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    #####################################################################
    # TODO:                                                             #
    # Tune learning rate and number of iterations. With the implemented #
    # code, report the validation and test accuracy.                    #
    #####################################################################
    lr = 0.01
    num_iteration = 200
    # train model
    theta, beta, val_acc_lst = irt(sparse_matrix, val_data, lr, num_iteration)
    # evaluate final accuracy
    print(f"train accuracy is {evaluate(train_data, theta, beta)}")
    print(f"valid accuracy is {evaluate(val_data, theta, beta)}")
    print(f"test accuracy is {evaluate(test_data, theta, beta)}")
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################

    #####################################################################
    # TODO:                                                             #
    # Implement part (d)                                                #
    #####################################################################
    j1, j2, j3 = 10, 100, 1000
    js = [j1, j2, j3]
    plt.figure()
    theta_x = np.linspace(0, max(theta) * 2, 100)
    for j in js:
        plt.plot(theta_x, sigmoid(theta_x - beta[j]), label="j = " + str(j))
    plt.xlabel(r"$\theta$")
    plt.ylabel("Probability")
    plt.title(r"Probability of Correctness vs. $\theta$")
    plt.legend()
    plt.show()
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


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


if __name__ == "__main__":
    main()
    # fig, ax1 = plt.subplots()
    # ax2 = ax1.twinx()
    #
    # ax1.plot(np.arange(200), lld_list, 'b')
    # ax2.plot(np.arange(200), val_lld_lst, 'r')
    # ax1.set_xlabel("#iteraion")
    # ax1.set_ylabel("negative log-likelihood")
    # ax2.set_ylabel("validation accuracy")
    # plt.show()
