import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties as font
import copy
from sklearn.model_selection import train_test_split
import imageio

# 生成輸入資料集(介於[-0.5, 0.2]或[0.8, 1.5]兩區塊之間)
def generate_input_data(size):
    input = []
    for i in range(size):
        if np.random.random() > 0.5:
            x = np.random.uniform(-0.5, 0.2)
        else:
            x = np.random.uniform(0.8, 1.5)
        
        if np.random.random() > 0.5:
            y = np.random.uniform(-0.5, 0.2)
        else:
            y = np.random.uniform(0.8, 1.5)
        
        input.append([x, y])
    
    return np.array(input)

#生成輸出資料集
def generate_output_data(input):
    output = []
    size = input.shape[0]
    for i in range(size):
        if ((input[i, 0] >= -0.5 and input[i, 0] <= 0.2) and (input[i, 1] >= 0.8 and input[i, 1] <= 1.5)) or ((input[i, 0] >= 0.8 and input[i, 0] <= 1.5) and (input[i, 1] >= -0.5 and input[i, 1] <= 0.2)):
            output.append(1)
        else:
            output.append(0)
    return np.array([output]).T

# 正規化輸入
def normalization_input(input):
    return (input + 0.5) / 2

# 反正規化輸出
def denormalization_output(output):
    if output <= 0.5:
        return 0
    else:
        return 1

#初始化權重
def initialize_weights(input_size, hidden_size, output_size):
    W1 = np.float64(np.random.randn(hidden_size, input_size))
    b1 = np.float64(np.zeros((hidden_size, 1)))
    W2 = np.float64(np.random.randn(output_size, hidden_size))
    return W1, b1, W2

#tangent sigmoid函數
def tangent_sigmoid(x):
    return np.tanh(x)

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

#散點圖
def draw_graf(epoch, testing_set_input, y_prediction, average_error):
    # 區分XOR結果為0(紅)與1的輸入(藍)
    input_red = []
    input_blue = []

    for i in range(len(y_prediction)):
        if y_prediction[i] == 0:
            input_red.append([testing_set_input[i, 0], testing_set_input[i, 1]])
        else:
            input_blue.append([testing_set_input[i, 0], testing_set_input[i, 1]])

    input_red = np.array(input_red).T
    input_blue = np.array(input_blue).T

    #繪製散點圖
    plt.figure()
    plt.grid(axis='both', ls='--', zorder=0)
    if input_red.shape[0] > 0:
        plt.scatter(input_red[0, :], input_red[1, :], color='red', s=4, label='XOR=0', zorder=10)
    if input_blue.shape[0] > 0:
        plt.scatter(input_blue[0, :], input_blue[1, :], color='blue', s=4, label='XOR=1',zorder=10)
    plt.xlim(-0.5,1.5)
    plt.ylim(-0.5,1.5)
    plt.text(0.9, 1.52,'結果平均誤差:%g' %average_error,{'fontsize':11},fontproperties=font(fname="font\\myfont.ttf"))
    plt.title('XOR結果分布圖', fontproperties=font(fname="font\\myfont.ttf"), fontsize=15)
    plt.legend(loc='center right', fontsize=10)
    plt.savefig('XOR結果分布圖(%d).png' %epoch)
    # plt.show()
    return

#訓練類神經網路
def train_neural_network(training_set_input, training_set_output, learning_rate, num_batch, W1, b1, W2):
    #記錄梯度
    dW1 = 0
    db1 = 0
    dW2 = 0

    for i in range(num_batch):
        for j in range(int(i*(training_set_input.shape[0]/num_batch)),int((i+1)*(training_set_input.shape[0]/num_batch))):
            x = training_set_input[j, :].reshape(-1, 1)
            y = training_set_output[j, 0].reshape(-1, 1)

            Z1, A1, Z2 = forward_propagation(x, W1, b1, W2)

            # # 反正規化輸出
            # Z2 = denormalization_output(Z2)

            dW1, db1, dW2 = backward_propagation(x, y, A1, Z1, Z2, W2, dW1, db1, dW2)
        
        dW1 = dW1 / (training_set_input.shape[0] / num_batch)
        db1 = db1 / (training_set_input.shape[0] / num_batch)
        dW2 = dW2 / (training_set_input.shape[0] / num_batch)

        W1, b1, W2 = update_parameters(W1, b1, W2, dW1, db1, dW2, learning_rate)

    return W1, b1, W2

#測試類神經網路
def test_neural_network(testing_set_input, testing_set_output, W1, b1, W2):
    #跑testing set
    y_prediction = []
    errors = []

    for i in range(testing_set_input.shape[0]):
        x = testing_set_input[i, :].reshape(-1, 1)

        y = testing_set_output[i, 0].reshape(-1, 1)

        Z2 = forward_propagation(x, W1, b1, W2)[2]

        # 反正規化輸出
        Z2 = denormalization_output(Z2)

        y_prediction.append(Z2)

        errors.append(np.square(Z2 - y[0]))

    # 誤差平均
    average_error = np.sqrt(np.mean(errors))

    return y_prediction, average_error

# 設置超參數(hyperparameters)
hidden_size = 2
times = 1
num_epoch = 150
num_batch = 10
learning_rate = 0.1

# 訓練不同隱藏層神經元數量的類神經網路(Train neural network for each hidden size)
for time in range(times):

    # 生成資料集
    input_data = generate_input_data(10000)
    output_data = generate_output_data(input_data)

    # 切割資料集
    training_set_input, testing_set_input, training_set_output, testing_set_output = train_test_split(input_data, output_data, test_size=0.2)

    #正規化輸入訓練集
    training_set_input_normalization = normalization_input(training_set_input)


    #正規化輸入驗證集
    testing_set_input_normalization = normalization_input(testing_set_input)

    #初始化權重
    W1, b1, W2 = initialize_weights(training_set_input.shape[1], hidden_size, 1)

    #未訓練時測試
    y_prediction, average_error = test_neural_network(testing_set_input_normalization, testing_set_output, W1, b1, W2)
    print('0', average_error)

    # 繪製散點圖
    draw_graf(0, testing_set_input, y_prediction, average_error)

    for epoch in range(num_epoch):
        W1, b1, W2 = train_neural_network(training_set_input_normalization, training_set_output, learning_rate, num_batch, W1, b1, W2)

        y_prediction, average_error = test_neural_network(testing_set_input_normalization, testing_set_output, W1, b1, W2)
        print(epoch, average_error)

        # 繪製散點圖
        draw_graf(epoch+1, testing_set_input, y_prediction, average_error)

# 初始化圖片列表
images = []

# 讀取每個epoch的圖片
for i in range(num_epoch + 1):
    images.append(imageio.imread(f'XOR結果分布圖({i}).png'))

# 創建GIF動畫
# 設置每幀持續50毫秒，即每秒顯示20幀
imageio.mimsave('XOR結果分布圖.gif', images, duration=50)
