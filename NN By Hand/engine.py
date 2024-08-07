import numpy as np


class Value:

    def __init__(self, data, _children=(),  _op='', label='', grad=0.0) -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
        self.grad = grad
        self._backward = lambda: None

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        out = self + (-other)
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __truediv__(self, other):
        out = self * other**-1
        return out

    def __radd__(self, other):
        return self + other

    def __rsub__(self, other):
        out = self + (-other)
        return out

    def __rmul__(self, other):
        out = self * other
        return out

    def __pow__(self, exponent):
        assert isinstance(exponent, (int, float)), "Only support int/float now"
        out = Value(self.data**exponent, (self, ), label=f'**{exponent}')

        def _backward():
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        out = Value(np.exp(self.data), (self, ), label='exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        n = self.data
        t = (np.exp(2*n) - 1) / (np.exp(2*n) + 1)
        out = Value(t, (self, ), label='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        n = self.data
        r = max(0, n)
        out = Value(r, (self, ), label='ReLU')

        def _backward():
            self.grad += r * out.grad
        out._backward = _backward
        return out

    def backward(self):

        # topological sort
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # perform back propagation
        # last node os the final output
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()
