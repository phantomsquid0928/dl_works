import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
# from common.gradient import numerical_gradient

class SimpleConvNet:
    """단순한 합성곱 신경망
    
    conv - relu - pool - affine - relu - affine - softmax
    
    Parameters
    ----------
    input_size : 입력 크기（MNIST의 경우엔 784）
    hidden_size_list : 각 은닉층의 뉴런 수를 담은 리스트（e.g. [100, 100, 100]）
    output_size : 출력 크기（MNIST의 경우엔 10）
    activation : 활성화 함수 - 'relu' 혹은 'sigmoid'
    weight_init_std : 가중치의 표준편차 지정（e.g. 0.01）
        'relu'나 'he'로 지정하면 'He 초깃값'으로 설정
        'sigmoid'나 'xavier'로 지정하면 'Xavier 초깃값'으로 설정
    """
    def __init__(self, input_dim=(1, 28, 28), 
                 conv_param={'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))

        conv_output_size2 = (pool_output_size - filter_size + 2 * filter_pad) / filter_stride + 1
        pool_output_size2 = int(filter_num * (conv_output_size2 / 2) * (conv_output_size2 / 2))
        
        #new init of weights
        #filter_num = output size
        self.conv_params = {'convd1' : {'filter_num' : 16, 'filter_size' : 3, 'pad' : 1, 'stride' : 1},
                       'convd2' : {'filter_num' : 32, 'filter_size' : 3, 'pad' : 1, 'stride' : 1},
                       'convu3' : {'filter_num' : 32, 'filter_size' : 3, 'pad' : 1, 'stride' : 1},
                       'convu4' : {'filter_num' : 16, 'filter_size' : 3, 'pad' : 1, 'stride' : 1},
                        'out' : {'filter_num' : output_size, 'filter_size' : 1, 'pad' : 0, 'stride' : 1}}
        channel_size = input_dim[0]
        input_size = input_dim[1]

        self.params = {}

        for name, vals in self.conv_params.items(): 
            cur_conv_output_size = (input_size - vals['filter_size'] + 2 * vals['pad']) / vals['stride'] + 1
            cur_pool_output_size = int(filter_num * (cur_conv_output_size / 2) * (cur_conv_output_size / 2))
            self.params['W_' + name] = weight_init_std * \
                                        np.random.randn(vals['filter_num'], channel_size, vals['filter_size'], vals['filter_size'])
            self.params['b_' + name] = np.zeros(vals['filter_num'])

            if name != 'out' :
                self.params['gamma_' + name] = np.zeros(cur_conv_output_size)
                self.params['beta_' + name] = np.zeros(cur_conv_output_size)
                channel_size = vals['filter_num']
                input_size = cur_pool_output_size
        # self.params['outW'] = weight_init_std * np.random.randn(self.outconv['filter_num'], channel_size, self.outconv['filter_size'], self.outconv['filter_size'])
        # self.params['outb'] = np.zeros(self.outconv['filter_num'])


        self.layers = OrderedDict()

        for name, vals in self.conv_params.items() :
            self.layers[name] = Convolution(self.params['W_' + name], self.params['b_' + name], vals['stride'], vals['pad'])
            if name != 'out' :
                self.layers['BatchNorm' + name] = BatchNormalization(self.params['gamma_' + name], self.params['beta_' + name])
                self.layers['relu_' + name] = Relu()
                self.layers['Pool_' + name] = Pooling(pool_h=2, pool_w=2, stride=2)
        
        # self.layers['output'] = Convolution(self.params['outW'], self.params['outb'], outconv['stride'], outconv['pad'])

            

        # 가중치 초기화
        # self.params = {}
        # self.params['W1'] = weight_init_std * \
        #                     np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        # self.params['b1'] = np.zeros(filter_num)
        
        # self.params['gamma1'] = np.zeros(conv_output_size)
        # self.params['beta1'] = np.zeros(conv_output_size)

        # self.params['W2'] = weight_init_std * \
        #                     np.random.randn(filter_num, filter_num, filter_size, filter_size)
        # self.params['b2'] = np.zeros(filter_num)


        # self.params['gamma2'] = np.zeros(conv_output_size2)
        # self.params['beta2'] = np.zeros(conv_output_size2)

        # self.params['W3'] = weight_init_std * \
        #                     np.random.randn(pool_output_size2, hidden_size)
        # self.params['b3'] = np.zeros(hidden_size)

        # self.params['W4'] = weight_init_std * \
        #                     np.random.randn(hidden_size, output_size)
        # self.params['b4'] = np.zeros(output_size)

        # 계층 생성
        # self.layers = OrderedDict()
        # self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
        #                                    conv_param['stride'], conv_param['pad'])
        
        # self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        # self.layers['Relu1'] = Relu()
        # self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        # self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'], 
        #                                    conv_param['stride'], conv_param['pad'])
        # self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        # self.layers['Relu2'] = Relu()
        # self.layers['Pool2'] = Pooling(pool_h = 2, pool_w = 2, stride = 2)

        # self.layers['Affine1'] = Affine(self.params['W3'], self.params['b3'])
        
        # self.layers['Relu2'] = Relu()
        # self.layers['Affine2'] = Affine(self.params['W4'], self.params['b4'])

        self.last_layer = SoftmaxWithLoss()
        # self.last_layer = BCELoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        # return x
        return Sigmoid(x)

    def loss(self, x, t):
        """손실 함수를 구한다.

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)
    
    # def calculate_vCDR(self, od, oc):
    #     """Calculate the vertical cup-to-disc ratio (vCDR) from the optic disc (od) and optic cup (oc) masks."""
    #     # Find the vertical diameter (sum along the y-axis)
    #     od_vertical_diameter = np.sum(od, axis=0).max()
    #     oc_vertical_diameter = np.sum(oc, axis=0).max()

    #     # Calculate the vertical cup-to-disc ratio
    #     vCDR = oc_vertical_diameter / (od_vertical_diameter + 1e-7)  # Add a small value to avoid division by zero
    #     return vCDR
    
    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            
            acc += softmax_loss(y, tt)
           # acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    # def numerical_gradient(self, x, t):
    #     """기울기를 구한다（수치미분）.

    #     Parameters
    #     ----------
    #     x : 입력 데이터
    #     t : 정답 레이블

    #     Returns
    #     -------
    #     각 층의 기울기를 담은 사전(dictionary) 변수
    #         grads['W1']、grads['W2']、... 각 층의 가중치
    #         grads['b1']、grads['b2']、... 각 층의 편향
    #     """
    #     loss_w = lambda w: self.loss(x, t)

    #     grads = {}
    #     for idx in (1, 2, 3):
    #         grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
    #         grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

    #     return grads

    def gradient(self, x, t):
        """기울기를 구한다(오차역전파법).

        Parameters
        ----------
        x : 입력 데이터
        t : 정답 레이블

        Returns
        -------
        각 층의 기울기를 담은 사전(dictionary) 변수
            grads['W1']、grads['W2']、... 각 층의 가중치
            grads['b1']、grads['b2']、... 각 층의 편향
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}

        for name in self.conv_params.keys():
            grads['W_' + name], grads['b_' + name] = self.layers[name].dW, self.layers[name].db
            if name != 'out' :
                grads['gamma_' + name], grads['beta_' + name] = self.layers['BatchNorm' + name].dgamma, self.layers['BatchNorm' + name].dbeta

        # grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        # grads['W2'], grads['b2'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        # grads['W3'], grads['b3'] = self.layers['Affine2'].dW, self.layers['Affine2'].db

        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]