#! usr/bin/env python3

# IMPORTS
from collections import Counter
from copy import deepcopy
import math
import random
import sys

# ACTIVATION FUNCTIONS
def linear(a):
    """Linear function"""
    return a

def sign(a):
    """Signum function"""
    if a > 0:
        return 1
    elif a < 0:
        return -1
    else:
        return 0

def hard_tanh(x):
    """Hard Hyperbolic tangent function """
    min_val = -1
    max_val = 1
    if x > max_val:
        return 1
    if x < min_val:
        return min_val
    else:
        return x

def sigmoid(a):
    """Sigmoid function"""
    if a < 0:
        return math.exp(a) / (1 + math.exp(a))
    else:
        return 1 / (1 + math.exp(-a))

def tanh(a):
    """Hyperbolic tangent function"""
    try:
        answer = (math.exp(a) - math.exp(-a)) / (math.exp(a) + math.exp(-a))
    except OverflowError:
        answer = sign(a)
    return answer

def softsign(a):
    """Softsign function"""
    return a / (1 + abs(a))

def relu(a):
    """Rectified linear unit function"""
    return max(0.0, a)

def softplus(a):
    """Softplus function"""
    return pseudo_ln(1 + math.exp(-abs(a))) + max(a, 0)


def nipuna(a, beta=1):
    """Nipuna function"""
    try:
        g_x = a / (1 + math.exp(-beta * a))
        answer = max(g_x, a)
    except OverflowError:
        answer = float('inf')
    return answer

def swish(a, beta=1):
    """Swish function"""
    return a * sigmoid(beta * a)


def hard_elish(x):
    """Hard elish function"""
    try:
        if x >= 1:
            y = x * max(0, min(1, ((x+1)/2)))
        if x < 0:
            y = (math.exp(x)-1)*max(0, min(1, ((x+1)/2)))
    except OverflowError:
        y = float('inf')
    return y

def elish(x):
    """Elish function"""
    try:
        if x < 0:
            y = (math.exp(x)-1)/(1+math.exp(-x))
        else:
            y = (x)/(1+math.exp(-x))
    except OverflowError:
        y = float('inf')
    return y


# LOSS FUNCTIONS
def mean_squared_error(yhat, y):
    """Mean squared loss function"""
    return (yhat - y) ** 2


def mean_absolute_error(yhat, y):
    """Absolute mean loss function"""
    return abs(yhat - y)


def hinge(yhat, y):
    """Hinge loss function"""
    return max(1 - yhat * y, 0)


def categorical_crossentropy(yhat, y):
    """Multinomial loss function"""
    return -y * pseudo_ln(yhat)


def binary_crossentropy(yhat, y):
    """Multinomial loss function"""
    return -y * pseudo_ln(yhat) - (1 - y) * pseudo_ln(1 - yhat)


# OTHER
def derivative(function, delta=1e-6):
    """Derivative of function"""

    def wrapper_derivative(x, *args):
        return (function(x + delta, *args) - function(x - delta, *args)) / (2.0 * delta)

    wrapper_derivative.__name__ = function.__name__ + '’'
    wrapper_derivative.__qualname__ = function.__qualname__ + '’'
    return wrapper_derivative


def pseudo_ln(x):
    """Pseudo-logarithmic function"""
    epsilon = 1e-10
    if x >= epsilon:
        return math.log(x)
    else:
        return math.log(epsilon) + ((x - epsilon) / epsilon)


# THE PERCEPTRON
class Perceptron:
    """Perceptron"""

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0 for x in range(dim)]

    def __repr__(self):
        text = f'Perceptron(dim={self.dim})'
        return text

    def predict(self, xs):
        """Predict y"""
        y = []
        for x in xs:
            y.append(self.get_activation(x))
        return y

    def partial_fit(self, xs, ys):
        """Adjust weight and bias"""
        for x, y in zip(xs, ys):
            yhat = self.get_activation(x)
            self.bias = self.bias - (yhat - y)
            for i in range(self.dim):
                self.weights[i] = self.weights[i] - ((yhat - y) * x[i])

    def fit(self, xs, ys, epochs=0):
        """Fit model"""
        if epochs == 0:
            change = True
            while change:
                b_bias = self.bias
                b_weights = self.weights[:]
                self.partial_fit(xs, ys)
                if b_bias == self.bias and b_weights == self.weights:
                    change = False
        else:
            for epoch in range(epochs):
                self.partial_fit(xs, ys)

    def get_activation(self, x):
        """Calculate activation"""
        a = self.bias
        for i in range(self.dim):
            a += self.weights[i] * x[i]
        return sign(a)

# LINEAR REGRESSION
class LinearRegression:
    """Linear Regression model"""

    def __init__(self, dim):
        self.dim = dim
        self.bias = 0.0
        self.weights = [0 for x in range(dim)]

    def __repr__(self):
        text = f'LinearRegression(dim={self.dim})'
        return text

    def predict(self, xs):
        """Predict y values"""
        return [self.get_yhat(x) for x in xs]

    def partial_fit(self, xs, ys, alpha=0.001):
        """Adjust weight and bias"""
        for x, y in zip(xs, ys):
            yhat = self.get_yhat(x)
            self.bias = self.bias - alpha * (yhat - y)
            for i in range(self.dim):
                self.weights[i] = self.weights[i] - alpha * (yhat - y) * x[i]

    def fit(self, xs, ys, alpha=0.1, epochs=50):
        """Fit model"""
        for epochs in range(epochs):
            self.partial_fit(xs, ys, alpha)

    def get_yhat(self, x):
        """Calculate yhat"""
        yhat = self.bias
        for i in range(self.dim):
            yhat += self.weights[i] * x[i]
        return yhat


# NEURON
class Neuron:
    """Neuron"""

    def __init__(self, dim, activation=linear, loss=mean_squared_error):
        self.dim = dim
        self.activation = activation
        self.loss = loss
        self.bias = 0.0
        self.weights = [0 for x in range(dim)]

    def __repr__(self):
        text = f'Neuron(dim={self.dim}, activation={self.activation.__name__}, loss={self.loss.__name__})'
        return text

    def predict(self, xs):
        """Predict y values"""
        yhats = []
        for instance in xs:
            a = self.get_activation(instance)
            yhat = self.activation(a)
            yhats.append(yhat)
        return yhats

    def partial_fit(self, xs, ys, alpha=0.001):
        """weights and bias updated by derived loss and activation functions"""
        for x, y in zip(xs, ys):
            a = self.get_activation(x)
            yhat = self.activation(a)
            loss_derivative = derivative(self.loss)
            activation_derivative = derivative(self.activation)
            self.bias = self.bias - alpha * loss_derivative(yhat, y) * activation_derivative(a)
            for i in range(self.dim):
                self.weights[i] = self.weights[i] - alpha * loss_derivative(yhat, y) * activation_derivative(a) * x[i]

    def fit(self, xs, ys, alpha=0.001, epochs=50):
        """Fit model"""
        for epochs in range(epochs):
            self.partial_fit(xs, ys, alpha)

    def get_activation(self, x):
        """Calculating activation by signum function"""
        a = self.bias
        for i in range(self.dim):
            a += self.weights[i] * x[i]
        return a


# LAYER
class Layer:
    """Layer"""

    class_counter = Counter()

    def __init__(self, outputs, *, name=None, next=None):
        Layer.class_counter[type(self)] += 1
        if name is None:
            name = f'{type(self).__name__}_{Layer.class_counter[type(self)]}'
        self.inputs = 0
        self.outputs = outputs
        self.name = name
        self.next = next

    def __repr__(self):
        text = f'Layer(inputs={self.inputs}, outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __add__(self, next):
        """Allow adding layers together"""
        result = deepcopy(self)
        result.add(deepcopy(next))
        return result

    def __getitem__(self, index):
        """Index layer"""
        if index == 0 or index == self.name:
            return self
        if isinstance(index, int):
            if self.next is None:
                raise IndexError('layer index out of range')
            return self.next[index - 1]
        if isinstance(index, str):
            if self.next is None:
                raise KeyError(index)
            return self.next[index]
        raise TypeError(f'layer indices must be integers or strings, not {type(index).__name__}')

    def __call__(self, xs):
        raise NotImplementedError('Abstract __call__ method')

    def add(self, next):
        """Add new layer and set output as new inputs"""
        if self.next is None:
            self.next = next
            next.set_inputs(self.outputs)
        else:
            self.next.add(next)

    def set_inputs(self, inputs):
        """Set inputs"""
        self.inputs = inputs


# INPUT LAYER
class InputLayer(Layer):
    """Input layer"""

    def __repr__(self):
        text = f'InputLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Call next layer"""
        return self.next(xs, ys, alpha)

    def predict(self, xs):
        """Pass predictions"""
        yhats, _, _ = self(xs)
        return yhats

    def evaluate(self, xs, ys):
        """Evaluate model"""
        _, ls, _ = self(xs, ys)
        return sum(ls) / len(ls)

    def partial_fit(self, xs, ys, alpha=0.001, batch_size=100):
        """Train model, return average loss"""
        if batch_size == 0:
            _, ls, _ = self(xs, ys, alpha)
        else:
            ls = []
            for i in range(0, len(xs), batch_size):
                _, batch_ls, _ = self(xs[i:i + batch_size], ys[i:i + batch_size], alpha)
                ls.extend(batch_ls)
        loss_avg = sum(ls) / len(ls)
        return loss_avg

    def fit(self, trn_xs, trn_ys, alpha=0.001, epochs=100, validation_data=([], []), batch_size=0):
        """Fit model, save the validation loss"""
        history = {'loss': []}

        if validation_data[0] and validation_data[1] != []:
            history['val_loss'] = []

        for e in range(epochs):
            training = list(zip(trn_xs, trn_ys))
            random.shuffle(training)
            trn_xs, trn_ys = zip(*training)
            train_loss_avg = self.partial_fit(trn_xs, trn_ys, alpha, batch_size)
            history['loss'].append(train_loss_avg)

            if validation_data[0] and validation_data[1] != []:
                val_loss_avg = self.evaluate(validation_data[0], validation_data[1])
                history['val_loss'].append(val_loss_avg)
        return history


# DENSE LAYER
class DenseLayer(Layer):
    """Dense layer"""

    def __init__(self, outputs, name=None):
        super().__init__(outputs)
        self.name = name
        self.bias = [0 for o in range(self.outputs)]
        self.weights = [[] for o in range(self.outputs)]

    def __repr__(self):
        text = f'DenseLayer(outputs={self.outputs}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Pre-activation with forward propagation, update bias and weights by backward propagation"""

        # Forward propagation
        aa = []
        for x in xs:
            a = []
            for o in range(self.outputs):
                pre_activation = sum(self.weights[o][i] * x[i] for i in range(self.inputs)) + self.bias[o]
                a.append(pre_activation)
            aa.append(a)

        # Backward propagation
        yhats, ls, gs = self.next(aa, ys, alpha)
        if ys and alpha is not None:
            grads = []
            for g, x in zip(gs, xs):
                gni = []
                for i in range(self.inputs):
                    gni_o = 0
                    for o in range(self.outputs):
                        gni_o += self.weights[o][i] * g[o]
                        self.bias[o] = self.bias[o] - (alpha / len(xs)) * g[o]
                        self.weights[o][i] = self.weights[o][i] - (alpha / len(xs)) * (g[o] * x[i])
                    gni.append(gni_o)
                grads.append(gni)
        else:
            grads = gs
        return yhats, ls, grads

    def set_inputs(self, inputs):
        """Set inputs and generate random weights"""
        self.inputs = inputs
        border = math.sqrt(6 / (self.inputs + self.outputs))
        for o in self.weights:
            for i in range(self.inputs):
                o.append(random.uniform(-border, border))


# ACTIVATION LAYER
class ActivationLayer(Layer):
    """Activation layer"""

    def __init__(self, outputs, activation=linear, name=None):
        super().__init__(outputs)
        self.name = name
        self.activation = activation
        self.d_activation = derivative(activation)

    def __repr__(self):
        text = f'ActivationLayer(outputs={self.outputs}, activation={self.activation.__name__}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Predict with pre-activation, adjust derivative with backward propagation"""
        # Forward propagation
        hh = []
        for x in xs:
            h = [self.activation(x[o]) for o in range(self.outputs)]
            hh.append(h)

        # Backward propagation
        yhats, ls, gs = self.next(hh, ys, alpha)
        if ys and alpha is not None:
            grads = []
            for x, g in zip(xs, gs):
                gnx = [self.d_activation(x[i]) * g[i] for i in range(self.inputs)]
                grads.append(gnx)
        else:
            grads = gs
        return yhats, ls, grads


# SOFTMAX LAYER
class SoftmaxLayer(Layer):
    """Softmax Layer"""

    def __init__(self, outputs):
        super().__init__(outputs)

    def __repr__(self):
        text = "SoftmaxLayer"
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Multinomial classification"""

        # Forward Propagation
        hh = []
        for x in xs:
            x = [v - max(x) for v in x]
            h = []
            ex_sum = sum(math.exp(x[i]) for i in range(self.inputs))
            for o in range(self.outputs):
                h.append(math.exp(x[o]) / ex_sum)
            hh.append(h)

        # Backward Propagation updating the gradient
        yhats, ls, gs = self.next(hh, ys, alpha)
        if ys and alpha is not None:
            grads = []
            for yhat, g in zip(yhats, gs):
                gnx = []
                for i in range(self.inputs):
                    gnx.append(sum(g[o] * yhat[o] * ((i == o) - yhat[i]) for o in range(self.outputs)))
                grads.append(gnx)
        else:
            grads = gs
        return yhats, ls, grads


# LOSS LAYER
class LossLayer(Layer):
    """Loss layer"""

    def __init__(self, loss=mean_squared_error, name=None):
        super().__init__(outputs=None)
        self.name = name
        self.loss = loss
        self.loss_derived = derivative(loss)

    def __repr__(self):
        text = f'LossLayer(loss={self.loss.__name__}, name={repr(self.name)})'
        if self.next is not None:
            text += ' + ' + repr(self.next)
        return text

    def __call__(self, xs, ys=None, alpha=None):
        """Calculate loss and adjust by backward propagation"""
        yhats = xs
        ls = None
        gs = None

        if ys is not None:
            ls = []
            for yhat, y in zip(yhats, ys):
                loss = sum(self.loss(yhat[i], y[i]) for i in range(self.inputs))
                ls.append(loss)

            if alpha is not None:
                gs = []
                for yhat, y in zip(yhats, ys):
                    g = [self.loss_derived(yhat[i], y[i]) for i in range(self.inputs)]
                    gs.append(g)
        return yhats, ls, gs

    def add(self, next):
        raise NotImplementedError("Loss is last Layer")


# MAIN
def main():
    return 0


if __name__ == "__main__":
    sys.exit(main())
