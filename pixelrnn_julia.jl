# This loads the MNIST handwritten digit recognition dataset:
include(Knet.dir("data","mnist.jl")) # Knet.dir constructs a path relative to Knet root
xtrn,ytrn,xtst,ytst = mnist()        # mnist() loads MNIST data and converts into Julia arrays
println.(summary.((xtrn,ytrn,xtst,ytst)));

# `minibatch` splits the data tensors to small chunks called minibatches.
# It returns a Knet.Data struct: an iterator of (x,y) pairs.
dtrn = minibatch(xtrn,ytrn,100)
dtst = minibatch(xtst,ytst,100)
