# This loads the MNIST handwritten digit recognition dataset:
using Pkg; installed = Pkg.installed()
for p in ("KnetLayers","Images","Statistics","Plots","IterTools","ImageMagick")
    haskey(installed,p) || Pkg.add(p)
end
using Base.Iterators: flatten
using IterTools: ncycle, takenth
using Statistics: mean
using KnetLayers, Images, Plots, Statistics, ImageMagick
using Knet: Knet, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data, sigmoid, sigm, minibatch
include(Knet.dir("data","mnist.jl")) # Knet.dir constructs a path relative to Knet root
#Pkg.update("Knet")
#Pkg.build("Knet")
xtrn,ytrn,xtst,ytst, = mnist()        # mnist() loads MNIST data and converts into Julia arrays
println.(summary.((xtrn,ytrn,xtst,ytst)));
#module list
#nvidia-smi
# `minibatch` splits the data tensors to small chunks called minibatches.
# It returns a Knet.Data struct: an iterator of (x,y) pairs.
dtrn = minibatch(xtrn,xtrn,100;xtype=KnetArray,ytype=KnetArray)
dtst = minibatch(xtst,xtst,100;xtype=KnetArray,ytype=KnetArray)

println(typeof(dtrn))

#(x,y) = first(dtrn)
#println.(summary.((x,y)))

function mask(inputs, num_outputs, kernelsize, mask_type)
kernel_h, kernel_w = kernelsize, kernelsize
height,width,channel,batch = inputs
#center_h = convert(Int32,floor(kernel_h / 2))
#center_w = convert(Int32,floor(kernel_w / 2))
center_h = convert(Int32,(kernel_h+1) / 2)
center_w = convert(Int32,(kernel_w+1) / 2)
#mask = param(kernel_h, kernel_w, channel, num_outputs)
mask = ones(Float32,(kernel_h, kernel_w, channel, num_outputs))
for j in center_w+1:kernel_w; mask[center_h,j, : , : ]=zeros(size(mask[center_h,j, : , : ]));end
for i in center_h+1:kernel_h; mask[i,: ,:,:]=zeros(size(mask[i,: ,:,:]));end
if mask_type == 'a'
    mask[center_h,center_w,:,:] = zeros(size(mask[center_h,center_w,:,:]))
end
return mask = KnetArray(mask.*randn(Float32,(kernel_h, kernel_w, channel, num_outputs)))
end

function softloss(ypred, ygold)
#    println(1,maximum(Array(ypred)),minimum(Array(ypred)))
#    println(2,maximum(Array(ygold)),minimum(Array(ygold)))
    ynorm = Array(ypred) .- log.(sum(exp.(Array(ypred))))
#    println(3,maximum(ynorm),minimum(ynorm))
    los = -sum(Array(ygold) .* ynorm)/(100*28*28)
#    println(4,maximum(los),minimum(los))
    return los
end


m1 = Param(mask((28,28,1,100), 16, 7, 'a'),SGD(lr=0.001));
m2 = Param(mask((28,28,16,100), 3, 3, 'b'),SGD(lr=0.001));
m3 = Param(mask((28,28,3,100), 3, 3, 'b'),SGD(lr=0.001));
m4 = Param(mask((28,28,3,100), 32, 1, 'b'),SGD(lr=0.001));
m5 = Param(mask((28,28,32,100), 32, 1, 'b'),SGD(lr=0.001));
m6 = Param(mask((28,28,32,100), 1, 1, 'b'),SGD(lr=0.001));

#main.m1 = mask(data, 16, 7, 'a');
#main.m2 = mask(conv_i, 3, 3, 'b');
#main.m3 = mask(outputs1, 3, 3, 'b');
#main.m4 = mask(outputs7, 32, 1, 'b');
#main.m5 = mask(outputs8, 32, 1, 'b');
#main.m6 = mask(outputs9, 1, [1, 'b');


function main(data,m1,m2,m3,m4,m5,m6)
data = KnetArray(data)
conv_i = Knet.batchnorm(conv4(m1,data, stride=1, padding=(3,3));training=true)
#println(0, typeof(conv_i))
#println(size(conv_i))

#first loop 7 times
outputs1 = Knet.batchnorm(conv4(m2,conv_i, stride=1, padding=(1,1));training=true)
#println(1, size(conv_i))
#println(typeof(outputs1),size(outputs1))

outputs2 = Knet.batchnorm(conv4(m3,outputs1, stride=1, padding=(1,1));training=true)
#println(2, size(outputs1))
#println(typeof(outputs2), size(outputs2))
outputs3 = Knet.batchnorm(conv4(m3,outputs2, stride=1, padding=(1,1));training=true)
#println(3, size(outputs2))
#println(typeof(outputs3), size(outputs3))
outputs4 = Knet.batchnorm(conv4(m3,outputs3, stride=1, padding=(1,1));training=true)
#println(4, size(outputs3))
#println(typeof(outputs4), size(outputs4))
outputs5 = Knet.batchnorm(conv4(m3,outputs4, stride=1, padding=(1,1));training=true)
#println(5, size(outputs4))
#println(typeof(outputs5), size(outputs5))
outputs6 = Knet.batchnorm(conv4(m3,outputs5, stride=1, padding=(1,1));training=true)
#println(6, size(outputs5))
#println(typeof(outputs6), size(outputs6))
outputs7 = Knet.batchnorm(conv4(m3,outputs6, stride=1, padding=(1,1));training=true)
#println(7, size(outputs6))
#println(typeof(outputs7), size(outputs7))

#second loop relu 2 times
temp8 = Knet.batchnorm(conv4(m4,outputs7, stride=1, padding=(0,0));training=true)
outputs8 = KnetArray(relu.(Array(temp8)))
#println(8, size(outputs7))
#println(typeof(outputs8), size(outputs8))

temp9 = Knet.batchnorm(conv4(m5,outputs8, stride=1, padding=(0,0));training=true)
outputs9 = KnetArray(relu.(Array(temp9)))
#println(9, size(outputs8))
#println(typeof(outputs9), size(outputs9))

#output sigmoid
conv2d_out_logits = Knet.batchnorm(conv4(m6,outputs9, stride=1, padding=(0,0));training=true)
#output = sigmoid.(Array(conv2d_out_logits))
#println(0, size(outputs9))
#println(typeof(output), size(output))
return conv2d_out_logits
end


for epoch in 1:100
epochloss=0
num=0
for (x,y) in dtrn
loss =@diff softloss(main(x,m1,m2,m3,m4,m5,m6),x)
epochloss=value(loss)+epochloss
num=num+1
lgm1 = grad(loss,m1)
update!(m1,lgm1)
lgm2 = grad(loss,m2)
update!(m2,lgm2)
lgm3 = grad(loss,m3)
update!(m3,lgm3)
lgm4 = grad(loss,m4)
update!(m4,lgm4)
lgm5 = grad(loss,m5)
update!(m5,lgm5)
lgm6 = grad(loss,m6)
update!(m6,lgm6)
end
l=epochloss/num
println(l)
end

#for epoch=1:100; train(main, dtrn); end
