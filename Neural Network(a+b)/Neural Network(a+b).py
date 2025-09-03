import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
from sklearn.model_selection import train_test_split

# 正規化輸入
def normalization_input(input):
    return np.float64((input - 1000) / 8999.0)

# 反正規化輸出
def denormalization_output(output):
    return output * 8999.0 + 1000

#tangent sigmoid函數
def tangent_sigmoid(x):
    return np.tanh(x)

#初始化權重
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.float64(np.random.randn(hidden_size, input_size))
    b1 = np.float64(np.zeros((hidden_size, 1)))
    W2 = np.float64(np.random.randn(output_size, hidden_size))

    return W1, b1, W2

#前向傳遞
def forward_propagation(x, W1, b1, W2):
    Z1 = np.dot(W1, x) + b1
    A1 = tangent_sigmoid(Z1)
    Z2 = np.dot(W2, A1)
    return Z1, A1, Z2

#反向傳遞
def backward_propagation(x, y, A1, Z1, Z2, W2, dW1, db1, dW2):
    error = y - Z2
    dW2 = dW2 - (error * A1.T)
    db1 = db1 - (error * W2.T * (4 / (np.square(np.exp(Z1) + np.exp(-Z1)))))
    dW1 = dW1 - (error * np.dot(W2.T * (4 / (np.square(np.exp(Z1) + np.exp(-Z1)))), x.T))
    return dW1, db1, dW2

#調整權重
def update_parameters(W1, b1, W2, dW1, db1, dW2, learning_rate):
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    return W1, b1, W2

#訓練類神經網路
def train_neural_network(training_set_input, training_set_output, learning_rate, num_batch, W1, b1, W2):
    #記錄梯度
    dW1 = 0
    db1 = 0
    dW2 = 0

    for i in range(num_batch):
        for j in range(int(i*(training_set_input.shape[0]/num_batch)),int((i+1)*(training_set_input.shape[0]/num_batch))):
            x = training_set_input[j, :].reshape(-1, 1)
            y = training_set_output[j].reshape(-1, 1)

            Z1, A1, Z2 = forward_propagation(x, W1, b1, W2)

            # 反正規化輸出
            Z2 = denormalization_output(Z2)

            dW1, db1, dW2 = backward_propagation(x, y, A1, Z1, Z2, W2, dW1, db1, dW2)

        dW1 = dW1 / (training_set_input.shape[0] / num_batch)
        db1 = db1 / (training_set_input.shape[0] / num_batch)
        dW2 = dW2 / (training_set_input.shape[0] / num_batch)

        W1, b1, W2 = update_parameters(W1, b1, W2, dW1, db1, dW2, learning_rate)

    return W1, b1, W2

#測試類神經網路
def test_neural_network(testing_set_input, testing_set_output, W1, b1, W2):
    errors = []

    for i in range(testing_set_input.shape[0]):
        x = testing_set_input[i, :].reshape(-1, 1)
        y = testing_set_output[i].reshape(-1, 1)

        Z2 = forward_propagation(x, W1, b1, W2)[2]

        # 反正規化輸出
        Z2 = denormalization_output(Z2)

        errors.append(np.square(Z2 - y[0]))

    # 誤差平均
    average_error = np.sqrt(np.mean(errors))

    return average_error

# 設置超參數(hyperparameters)
hidden_size_list = list(range(1, 21))
times = 30
num_epoch = 100
num_batch = 10
learning_rate = 0.00002

# 存放不同數量神經元訓練後的權重與誤差平均
average_error_list = np.zeros((times, len(hidden_size_list)))

# 訓練不同隱藏層神經元數量的類神經網路(Train neural network for each hidden size)
for time in range(times):

    # 生成資料集
    input_data = np.random.randint(1000, 10000, size=(10000, 2))
    output_data = np.sum(input_data, axis=1)

    # 切割資料集
    training_set_input, testing_set_input, training_set_output, testing_set_output = train_test_split(input_data, output_data, test_size=0.2)

    #正規化輸入訓練集
    training_set_input = normalization_input(training_set_input)


    #正規化輸入驗證集
    testing_set_input = normalization_input(testing_set_input)


    for hidden_size in hidden_size_list:
        #初始化權重
        W1, b1, W2 = initialize_weights(training_set_input.shape[1], hidden_size, 1)

        for epoch in range(num_epoch):
            W1, b1, W2 = train_neural_network(training_set_input, training_set_output, learning_rate, num_batch, W1, b1, W2)
            
        average_error = test_neural_network(testing_set_input, testing_set_output, W1, b1, W2)
        print(time, hidden_size, average_error)

        average_error_list[time, hidden_size-1] = average_error

average_error_list = np.sum(average_error_list, axis=0) / times

# 神經元數量對30次訓練結果之誤差平均圖(折線圖)
plt.figure(figsize=(14, 2), dpi=100)
plt.grid(axis='y', ls='--', zorder=0)
plt.plot(hidden_size_list, list(np.floor(np.array(average_error_list) * 100) / 100.0), color='#375da1', marker='o', zorder=10)
plt.ylim(0,max(list(np.floor(np.array(average_error_list) * 100) / 100.0))+100)
plt.xticks(hidden_size_list, [str(num) for num in hidden_size_list])
plt.title('神經元數量對30次訓練結果之誤差平均圖', fontproperties=font(fname="font\\myfont.ttf"), fontsize=15)
for i in range(len(average_error_list)):
    plt.text(hidden_size_list[i]-0.4,list(np.floor(np.array(average_error_list) * 100) / 100.0)[i]+50,list(np.floor(np.array(average_error_list) * 100) / 100.0)[i],fontsize=7,zorder=20)
plt.savefig('神經元數量對30次訓練結果之誤差平均圖.png', bbox_inches='tight')
plt.show()