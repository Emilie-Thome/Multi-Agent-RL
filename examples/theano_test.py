import theano
import numpy
 
x = theano.tensor.fvector('x')
target = theano.tensor.fscalar('target')
 
W = theano.shared(numpy.asarray([0.2, 0.7]), 'W')
y = (x * W).sum()
 
cost = theano.tensor.sqr(target - y)
gradient = theano.tensor.grad(cost, [W]) 
f = theano.function(inputs=[x, target], outputs=gradient, on_unused_input='ignore')

for i in range(10):
    output = f([1.0, 1.0], 20.0)
    print(output)