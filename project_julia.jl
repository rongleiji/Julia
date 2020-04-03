# This loads the MNIST handwritten digit recognition dataset:
using Knet: Knet, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data, sigmoid, sigm, minibatch
include(Knet.dir("data","mnist.jl")) # Knet.dir constructs a path relative to Knet root
Pkg.update("Knet")
Pkg.build("Knet")
xtrn,ytrn,xtst,ytst, = mnist()        # mnist() loads MNIST data and converts into Julia arrays
println.(summary.((xtrn,ytrn,xtst,ytst)));
#module list
#nvidia-smi
# `minibatch` splits the data tensors to small chunks called minibatches.
# It returns a Knet.Data struct: an iterator of (x,y) pairs.
dtrn = minibatch(xtrn,ytrn,100)
dtst = minibatch(xtst,ytst,100)
#dtrn,dtst = mnistdata(1)

println(typeof(dtrn))

function mask(inputs, num_outputs, kernelsize, mask_type)
kernel_h, kernel_w = kernelsize
height,width,channel,batch = size(inputs[1])
#center_h = convert(Int32,floor(kernel_h / 2))
#center_w = convert(Int32,floor(kernel_w / 2))
center_h = convert(Int32,(kernel_h+1) / 2)
center_w = convert(Int32,(kernel_w+1) / 2)
mask = ones(Float32,(kernel_h, kernel_w, channel, num_outputs))
for j in center_w+1:kernel_w; mask[center_h,j, : , : ]=zeros(size(mask[center_h,j, : , : ]));end
for i in center_h+1:kernel_h; mask[i,: ,:,:]=zeros(size(mask[i,: ,:,:]));end
if mask_type == 'a'
    mask[center_h,center_w,:,:] = zeros(size(mask[center_h,center_w,:,:]))
end
return mask .*= randn(kernel_h, kernel_w)
end

dx = []
for (x,y) in dtrn
push!(dx,x)
end
println(typeof(dx))
println(size(dx))
m1 = mask(dx, 16, [7 7], 'a');
conv_i = []
for (x,y) in dtrn
push!(conv_i,conv4(m1,x, stride=1, padding=(3,3)))
end
println(typeof(conv_i))
println(size(conv_i))
println(typeof(conv_i[1]))

#first loop 7 times
m2 = mask(conv_i, 3, [3 3], 'b');
outputs1=Any[]
for x in conv_i
push!(outputs1,conv4(m2,x, stride=1, padding=(1,1)))
end
println(size(conv_i[1]))
println(1, typeof(outputs1[1]), size(outputs1[1]))

m3 = mask(outputs1, 3, [3 3], 'b');
outputs2=Any[]
for x2 in outputs1
push!(outputs2,conv4(m3,x2, stride=1, padding=(1,1)))
end
println(size(outputs1[1]))
println(2, typeof(outputs2[1]), size(outputs2[1]))
outputs3=Any[]
for x3 in outputs2
push!(outputs3,conv4(m3,x3, stride=1, padding=(1,1)))
end
println(size(outputs2[1]))
println(3, typeof(outputs3[1]), size(outputs3[1]))
outputs4=Any[]
for x4 in outputs3
push!(outputs4,conv4(m3,x4, stride=1, padding=(1,1)))
end
println(size(outputs3[1]))
println(4, typeof(outputs4[1]), size(outputs4[1]))
outputs5=Any[]
for x5 in outputs4
push!(outputs5,conv4(m3,x5, stride=1, padding=(1,1)))
end
println(size(outputs4[1]))
println(5, typeof(outputs5[1]), size(outputs5[1]))
outputs6=Any[]
for x6 in outputs5
push!(outputs6,conv4(m3,x6, stride=1, padding=(1,1)))
end
println(size(outputs5[1]))
println(6, typeof(outputs6[1]), size(outputs6[1]))
outputs7=Any[]
for x7 in outputs6
push!(outputs7,conv4(m3,x7, stride=1, padding=(1,1)))
end
println(size(outputs6[1]))
println(7, typeof(outputs7[1]), size(outputs7[1]))

#second loop relu 2 times
m4 = mask(outputs7, 32, [1 1], 'b');
outputs8=Any[]
for x8 in outputs7
push!(outputs8,relu.(conv4(m4,x8, stride=1, padding=(0,0))))
end
println(size(outputs7[1]))
println(8, typeof(outputs8[1]), size(outputs8[1]))

m5 = mask(outputs8, 32, [1 1], 'b');
outputs9=Any[]
for x9 in outputs8
push!(outputs9,relu.(conv4(m5,x9, stride=1, padding=(0,0))))
end
println(size(outputs8[1]))
println(9, typeof(outputs9[1]), size(outputs9[1]))

#output sigmoid
m6 = mask(outputs9, 1, [1 1], 'b');
conv2d_out_logits=Any[]
for x0 in outputs9
push!(conv2d_out_logits,conv4(m6,x0, stride=1, padding=(1,1)))
end
output = Any[]
for x in conv2d_out_logits
push!(output, sigmoid.(x))
end
println(size(outputs9[1]))
println(0, typeof(output[1]), size(output[1]))