import numpy as np
from collections import OrderedDict
from common.layers import *
import pickle

class BatchNormWrapper:
    def __init__(self, batch_norm_layer):
        self.batch_norm_layer = batch_norm_layer

    def forward(self, x):
        if x.ndim == 4:
            N, C, H, W = x.shape
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
            x_flat = self.batch_norm_layer.forward(x_flat)
            x = x_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            return x
        else:
            return self.batch_norm_layer.forward(x)

    def backward(self, dout):
        if dout.ndim == 4:
            N, C, H, W = dout.shape
            dout_flat = dout.transpose(0, 2, 3, 1).reshape(-1, C)
            dout_flat = self.batch_norm_layer.backward(dout_flat)
            dout = dout_flat.reshape(N, H, W, C).transpose(0, 3, 1, 2)
            return dout
        else:
            return self.batch_norm_layer.backward(dout)

class Upsample:
    def __init__(self, scale_factor=2):
        self.scale_factor = scale_factor

    def forward(self, x):
        return np.repeat(np.repeat(x, self.scale_factor, axis=2), self.scale_factor, axis=3)

    def backward(self, dout):
        N, C, H, W = dout.shape
        return dout[:, :, ::self.scale_factor, ::self.scale_factor]

class Concat:
    """Concatenate two inputs along the channel axis."""
    def forward(self, x1, x2):
        self.x1_shape = x1.shape
        self.x2_shape = x2.shape
        return np.concatenate([x2, x1], axis=1)

    def backward(self, dout):
        dout_x1 = dout[:, self.x2_shape[1]:, :, :]  # grad for x1 (original input)
        dout_x2 = dout[:, :self.x2_shape[1], :, :]  # grad for x2 (upsampled input)
        return dout_x1, dout_x2

class DoubleConv: #now will do only 1 as we need lighter one
    def __init__(self, W1, b1, stride=1, pad=1):
        self.conv1 = Convolution(W1, b1, stride=stride, pad=pad)
        self.bn1 = BatchNormWrapper(BatchNormalization(np.ones(W1.shape[0]), np.zeros(W1.shape[0])))
        self.relu1 = Relu()
        # self.conv2 = Convolution(W2, b2, stride=stride, pad=pad)
        # self.bn2 = BatchNormWrapper(BatchNormalization(np.ones(W2.shape[0]), np.zeros(W2.shape[0])))
        # self.relu2 = Relu()

    def forward(self, x):
        x = self.conv1.forward(x)
        x = self.bn1.forward(x)
        x = self.relu1.forward(x)
        # x = self.conv2.forward(x)
        # x = self.bn2.forward(x)
        # x = self.relu2.forward(x)
        return x

    def backward(self, dout):
        # dout = self.relu2.backward(dout)
        # dout = self.bn2.backward(dout)
        # dout = self.conv2.backward(dout)
        dout = self.relu1.backward(dout)
        dout = self.bn1.backward(dout)
        dout = self.conv1.backward(dout)
        return dout

    @property
    def dW1(self):
        return self.conv1.dW

    @property
    def db1(self):
        return self.conv1.db

    # @property
    # def dW2(self):
    #     return self.conv2.dW

    # @property
    # def db2(self):
    #     return self.conv2.db

    @property
    def dgamma1(self):
        return self.bn1.dgamma

    @property
    def dbeta1(self):
        return self.bn1.dbeta

    # @property
    # def dgamma2(self):
    #     return self.bn2.dgamma

    # @property
    # def dbeta2(self):
    #     return self.bn2.dbeta

class Down:
    """Double Conv followed by Max Pooling for downsampling."""
    def __init__(self, W1, b1, stride=1, pad=1):
        self.double_conv = DoubleConv(W1, b1, stride, pad)
        self.pool = Pooling(pool_h=2, pool_w=2, stride=2)
        # self.saved_dout = None  # Initialize saved dout for gradient accumulation

    def forward(self, x):
        conv = self.double_conv.forward(x)
        x = self.pool.forward(conv)  # Apply max pooling after the double conv
        return x, conv

    def backward(self, dout, dout2):
        # if add and self.saved_dout is not None:
        #     print(f'saved shape {self.save_dout.shape}')
        #     print(f'dout shape {dout.shape}')
        #     # If add=True, sum the current dout and saved_dout element-wise
        #     dout += self.saved_dout  # Element-wise addition, dimensions must match
        dout = self.pool.backward(dout)  # Max pooling's backward
        dout += dout2
        dout = self.double_conv.backward(dout)  # Backprop through double conv
        return dout

    # def save_dout(self, dout):
    #     # Save dout for future gradient accumulation, ensure it's saved correctly
    #     self.saved_dout = dout

    # Access gradients for backpropagation
    @property
    def dW1(self):
        return self.double_conv.dW1

    @property
    def db1(self):
        return self.double_conv.db1

    # @property
    # def dW2(self):
    #     return self.double_conv.dW2

    # @property
    # def db2(self):
    #     return self.double_conv.db2

    @property
    def dgamma1(self):
        return self.double_conv.dgamma1

    @property
    def dbeta1(self):
        return self.double_conv.dbeta1

    # @property
    # def dgamma2(self):
    #     return self.double_conv.dgamma2

    # @property
    # def dbeta2(self):
    #     return self.double_conv.dbeta2


class Up:
    """Upsample followed by concatenation and Double Conv."""
    def __init__(self, W1, b1, stride=1, pad=1):
        self.up = Upsample(scale_factor=2)
        self.concat = Concat()
        self.double_conv = DoubleConv(W1, b1, stride, pad)
        self.x2 = False
    def forward(self, x1, x2=None):
        x1 = self.up.forward(x1)
        if x2 is not None:
            # print(f'before concat size x1 x2 - x1 : {x1.shape} ,  x2 :  {x2.shape}')
            x1 = self.concat.forward(x1, x2)
            self.x2 = True
        return self.double_conv.forward(x1)

    def backward(self, dout):
        dout = self.double_conv.backward(dout)
        if self.x2 is True:
            dout_x1, dout_x2 = self.concat.backward(dout)
            dout_x1 = self.up.backward(dout_x1)
            # print(f'shape dout_x1{dout_x1.shape}')
            # print(f'shape dout_x2{dout_x2.shape}')
            return dout_x1, dout_x2
        return dout

    @property
    def dW1(self):
        return self.double_conv.dW1

    @property
    def db1(self):
        return self.double_conv.db1

    # @property
    # def dW2(self):
    #     return self.double_conv.dW2

    # @property
    # def db2(self):
    #     return self.double_conv.db2

    @property
    def dgamma1(self):
        return self.double_conv.dgamma1

    @property
    def dbeta1(self):
        return self.double_conv.dbeta1

    # @property
    # def dgamma2(self):
    #     return self.double_conv.dgamma2

    # @property
    # def dbeta2(self):
    #     return self.double_conv.dbeta2

class SimpleConvNet:
    def __init__(self, input_dim=(3, 256, 256), output_size=2, weight_init_std=0.01):
        self.conv_params = {
            'convd1': {'filter_in': input_dim[0], 'filter_out' : 16, 'filter_size': 3, 'pad': 1, 'stride': 1},
            'convd2': {'filter_in': 16, 'filter_out' : 32,'filter_size': 3, 'pad': 1, 'stride': 1},
            'conv3': {'filter_in': 32, 'filter_out' : 64,'filter_size': 3, 'pad': 1, 'stride': 1},
            'convu1': {'filter_in': 96, 'filter_out' : 48,'filter_size': 3, 'pad': 1, 'stride': 1},
            'convu2': {'filter_in': 64, 'filter_out' : 32,'filter_size': 3, 'pad': 1, 'stride': 1},
            'out': {'filter_in': 32, 'filter_out' : output_size, 'filter_size': 1, 'pad': 0, 'stride': 1}
        }
        self.params = {}
        self.layers = OrderedDict()

        input_size = input_dim[1]

        for name, vals in self.conv_params.items():
            cur_conv_output_size = (input_size - vals['filter_size'] + 2 * vals['pad']) / vals['stride'] + 1
            cur_pool_output_size = int(cur_conv_output_size / 2)

            if name == 'out':
                # Only single convolution for output
                #added he init
                W = np.random.randn(vals['filter_out'], vals['filter_in'], vals['filter_size'], vals['filter_size']) * np.sqrt(2.0 / vals['filter_in'])
                b = np.zeros(vals['filter_out'])
                self.params[f'W_{name}'] = W
                self.params[f'b_{name}'] = b
                self.layers[name] = Convolution(W, b, vals['stride'], vals['pad'])
            else:
                W1 = np.random.randn(vals['filter_out'], vals['filter_in'], vals['filter_size'], vals['filter_size']) * np.sqrt(2.0 / vals['filter_in'])
                # W2 = weight_init_std * np.random.randn(vals['filter_out'], vals['filter_out'], vals['filter_size'], vals['filter_size'])
                b1 = np.zeros(vals['filter_out'])
                # b2 = np.zeros(vals['filter_out'])

                self.params[f'W1_{name}'] = W1
                self.params[f'b1_{name}'] = b1
                # self.params[f'W2_{name}'] = W2
                # self.params[f'b2_{name}'] = b2

                if 'convd' in name:
                    self.layers[name] = Down(W1, b1)
                elif 'convu' in name:
                    self.layers[name] = Up(W1, b1)
                elif 'conv' in name:
                    self.layers[name] = DoubleConv(W1, b1)

            input_size = cur_pool_output_size


        self.last_layer = BCELoss()

    def forward(self, x):
        enc1, cres1 = self.layers['convd1'].forward(x)
        # print(f'convd1 res shape : {enc1.shape}   - saved shape : {cres1.shape}')
        enc2, cres2 = self.layers['convd2'].forward(enc1)
        # print(f'convd2 res shape : {enc2.shape}     - saved shape : {cres2.shape}')
        enc3 = self.layers['conv3'].forward(enc2)
        # print(f'conv3 res shape : {enc3.shape}')

        dec1 = self.layers['convu1'].forward(enc3, cres2)
        # print(f'convu1 res shape : {dec1.shape}')
        dec2 = self.layers['convu2'].forward(dec1, cres1)
        # print(f'convu2 res shape : {dec2.shape}')

        out = self.layers['out'].forward(dec2)
        # print(f'out res shape : {out.shape}')
        print(f"Model output range (before sigmoid): min={out.min()}, max={out.max()}")
        return sigmoid(out)

    def predict(self, x):
        return self.forward(x)

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        total_iou_od = 0  # For optic disc
        total_iou_oc = 0  # For optic cup
        total_batches = 0

        for i in range(0, x.shape[0], batch_size):
            print(f'acc assess {i}...')
            x_batch = x[i:i + batch_size]
            t_batch = t[i:i + batch_size]

            # Get predictions for the batch
            y_batch = self.predict(x_batch)

            # Threshold to get binary predictions (0 or 1) for both channels
            y_bin = (y_batch > 0.5).astype(np.float32)
            t_bin = t_batch.astype(np.float32)

            # Optic disc IoU (first channel)
            intersection_od = np.sum((y_bin[:, 0] == 1) & (t_bin[:, 0] == 1))  # True positives
            union_od = np.sum((y_bin[:, 0] == 1) | (t_bin[:, 0] == 1))         # True positives + False positives + False negatives
            iou_od = intersection_od / (union_od + 1e-7)

            # Optic cup IoU (second channel)
            intersection_oc = np.sum((y_bin[:, 1] == 1) & (t_bin[:, 1] == 1))  # True positives
            union_oc = np.sum((y_bin[:, 1] == 1) | (t_bin[:, 1] == 1))         # True positives + False positives + False negatives
            iou_oc = intersection_oc / (union_oc + 1e-7)

            total_iou_od += iou_od
            total_iou_oc += iou_oc
            total_batches += 1

        # Return average IoU for both optic disc and optic cup
        avg_iou_od = total_iou_od / total_batches
        avg_iou_oc = total_iou_oc / total_batches
        avg_iou = (avg_iou_od + avg_iou_oc) / 2.0  # Average over both classes
        return avg_iou




    def gradient(self, x, t):
      """Compute the gradient using backpropagation."""
      # Forward pass to calculate the loss
      self.loss(x, t)
      dout = self.last_layer.backward(1)

      # Reverse the layers for backward propagation
      # rlayers = sorted(self.layers., reverse = True)
      
      reverselayer = ['out', 'convu2', 'convu1', 'conv3', 'convd2', 'convd1']
      # print(f'sorted res : {reverselayer}')
      saved_douts = {}  # To store saved gradients for concatenation

      # Backward pass through all layers

      layermap = {'convd2' : 'convu1', 'convd1' : 'convu2' }
      for name in reverselayer:
          layer = self.layers[name]
          # print(f'shape of dout {dout.shape}')
          # print(f'back proping : {name} layer...')
          # print()
          # If the layer is an Up layer with concatenation
          if isinstance(layer, Up):
              # Perform the backward pass with concatenation
              dout, dout_x2 = layer.backward(dout)
              # Save dout_x2 for the corresponding Down layer
              saved_douts[name] = dout_x2
          elif isinstance(layer, Down):
              # If it's a Down layer and it has saved dout from a concatenated Up layer
              dout = layer.backward(dout, saved_douts[layermap[name]])  # Combine the saved dout with the current dout

          else:
              # Regular backward for other layers
              dout = layer.backward(dout)

      # Collect gradients
      grads = {}
      for name, layer in self.layers.items():
        if isinstance(layer, (Down, Up, DoubleConv)):
            # Collect gradients for layers with DoubleConv (including Down, Up, and DoubleConv itself)
          grads[f'W1_{name}'] = layer.dW1
          grads[f'b1_{name}'] = layer.db1
        #   grads[f'W2_{name}'] = layer.dW2
        #   grads[f'b2_{name}'] = layer.db2
          if hasattr(layer, 'dgamma1'):
              grads[f'gamma1_{name}'] = layer.dgamma1
              grads[f'beta1_{name}'] = layer.dbeta1
        #   if hasattr(layer, 'dgamma2'):
        #       grads[f'gamma2_{name}'] = layer.dgamma2
        #       grads[f'beta2_{name}'] = layer.dbeta2
        elif isinstance(layer, Convolution):
          # Collect gradients for individual Convolution layers
          grads[f'W_{name}'] = layer.dW
          grads[f'b_{name}'] = layer.db


      return grads

import pickle

class SimpleConvNet:
    # Other methods and initialization as before...

    def save_params(self, file_name="params.pkl"):
        """ Save the model parameters into a pickle file """
        params = {}
        for key, val in self.params.items():
            params[key] = val
        
        # Add BatchNorm parameters to params
        for key, layer in self.layers.items():
            if isinstance(layer, (Down, Up, DoubleConv)):
                params[f'gamma1_{key}'] = layer.bn1.batch_norm_layer.gamma
                params[f'beta1_{key}'] = layer.bn1.batch_norm_layer.beta
                if hasattr(layer, 'bn2'):  # In case there's a second BN layer (for double convs)
                    params[f'gamma2_{key}'] = layer.bn2.batch_norm_layer.gamma
                    params[f'beta2_{key}'] = layer.bn2.batch_norm_layer.beta

        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

def load_params(self, file_name="params.pkl"):
    """ Load model parameters from a pickle file and assign to layers """
    with open(file_name, 'rb') as f:
        loaded_params = pickle.load(f)

    # Update self.params with loaded parameters
    for key, val in loaded_params.items():
        self.params[key] = val

    # Assign the loaded weights, biases, and batchnorm params back to the corresponding layers
    for key, layer in self.layers.items():
        if isinstance(layer, (Down, Up, DoubleConv)):  # Layers with weights
            layer.W1 = self.params[f'W1_{key}']
            layer.b1 = self.params[f'b1_{key}']
            if hasattr(layer, 'W2'):
                layer.W2 = self.params[f'W2_{key}']
                layer.b2 = self.params[f'b2_{key}']
            
            # Load BatchNorm parameters
            layer.bn1.batch_norm_layer.gamma = self.params[f'gamma1_{key}']
            layer.bn1.batch_norm_layer.beta = self.params[f'beta1_{key}']
            if hasattr(layer, 'bn2'):
                layer.bn2.batch_norm_layer.gamma = self.params[f'gamma2_{key}']
                layer.bn2.batch_norm_layer.beta = self.params[f'beta2_{key}']
        elif isinstance(layer, Convolution):  # For the 'out' layer
            layer.W = self.params[f'W_{key}']
            layer.b = self.params[f'b_{key}']

