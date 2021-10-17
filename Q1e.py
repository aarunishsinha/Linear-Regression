import os
import sys
import numpy as np
import pandas as pd
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy import linalg
from decimal import Decimal
matplotlib.use('Agg')
def initialise(x,y):
    x_csv = x 
    y_csv = y 
    x_dataf = pd.read_csv(x_csv, header = None)
    y_dataf = pd.read_csv(y_csv, header = None)
    x_train = (np.array(x_dataf))
    mean = np.mean(x_train,axis=0)
    var = np.var(x_train, axis =0)
    x_train -= mean
    np.divide(x_train, var)
    y_train = (np.array(y_dataf))
    x_zero = 1.0 + np.zeros((np.shape(x_train)[0], 1), dtype = float)
    x_train = np.hstack((x_train, x_zero))
    return x_train, y_train, x_dataf, y_dataf
def cost_func(theta, x_train, y_train):
    hypothesis = np.dot(x_train, theta)
    m = np.shape(y_train)[0]
    cost = (1/(2*m))*np.sum(((y_train - hypothesis)**2), axis = 0)
    return cost[0]
def gradient(theta, x_train, y_train):
    m = np.shape(y_train)[0]
    grad = np.zeros((2,1))
    grad[0] = sum((y_train - np.dot(x_train, theta))*x_train[:,0][:,None])
    grad[1] = sum((y_train - np.dot(x_train, theta))*x_train[:,1][:,None])
    grad = -1*grad/m
    return grad
def gradient_descent(theta, learning_rate, x_train, y_train):
    iteration = 0
    epsilon = 0.0000000001 
    new_cost = 0
    thetas = [[0,0]]
    old_cost = cost_func(theta, x_train, y_train)
    costs = [old_cost]
    while True:
        grad = gradient(theta, x_train, y_train)
        theta -= learning_rate*grad
        new_cost = cost_func(theta, x_train, y_train)
        costs = np.append(costs,[new_cost], axis = 0)
        thetas = np.append(thetas, [[theta[0][0], theta[1][0]]],axis = 0)
        del_cost = old_cost - new_cost
        if del_cost<epsilon:
            break
        old_cost = new_cost
        iteration += 1
    return theta, new_cost, thetas, costs
def plot4(time_gap, x_train, y_train, thetas, costs, iterr):
    plt.figure()
    l = max(thetas[-1][0], thetas[-1][1])
    x1 = np.linspace(-2*l, 2*l, 100)
    x2 = np.linspace(-2*l, 2*l, 100)
    space = np.meshgrid(x1,x2)
    Z = np.zeros((100,100), dtype = float)
    for i in range(100):
        for j in range(100):
            Z[i][j] = cost_func([[space[0][i][j]], [space[1][i][j]]], x_train, y_train)
    surf = plt.contour(space[0], space[1], Z)
    plt.xlabel('theta_1')
    plt.ylabel('theta_0')
    plt.ion()
    for i in range(np.shape(thetas)[0]):
        plt.scatter(thetas[i][0], thetas[i][1], costs[i], color = 'r')
        # plt.pause(time_gap)
    plt.ioff()
    plt.savefig(os.path.join(sys.argv[2], str(iterr) + 'q1_e.png' ),dpi=200)
    # plt.show()
def main(eta):
    x = os.path.join(sys.argv[1], 'linearX.csv')
    y = os.path.join(sys.argv[1], 'linearY.csv')
    (x_train, y_train, x_dataf, y_dataf) = initialise(x,y)
    theta = np.zeros((np.shape(x_train)[1], 1), dtype = float)
    time_gap = 0.2
    iterr = 1
    write_file = open(os.path.join(sys.argv[2], "q1_e.txt"), "w")
    for learning_rate in eta:
        (Theta, final_cost, thetas, costs) = gradient_descent(theta, learning_rate, x_train, y_train)
        theta = np.zeros((np.shape(x_train)[1], 1), dtype = float)
        write_file.write("Learning Rate = ")
        write_file.write(str(learning_rate))
        write_file.write('\n')
        write_file.write("Theta_1 = ")
        write_file.write(str(Theta[0][0]))
        write_file.write('\n')
        write_file.write("Theta_0 = ")
        write_file.write(str(Theta[1][0]))
        write_file.write('\n')
        plot4(time_gap, x_train, y_train, thetas, costs, iterr)
        iterr += 1
    write_file.write("As the learning rate increases the number of iterations in gradient descent decreases such that after a certain value the value of descent step gets so large that the it never converges.")
if __name__ == "__main__":
    main([0.001, 0.025, 0.1])