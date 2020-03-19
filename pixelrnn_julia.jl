# This loads the MNIST handwritten digit recognition dataset:
using Knet
include(Knet.dir("data","mnist.jl")) # Knet.dir constructs a path relative to Knet root
xtrn,ytrn,xtst,ytst, = mnist()        # mnist() loads MNIST data and converts into Julia arrays
println.(summary.((xtrn,ytrn,xtst,ytst)));

# `minibatch` splits the data tensors to small chunks called minibatches.
# It returns a Knet.Data struct: an iterator of (x,y) pairs.
dtrn = minibatch(xtrn,ytrn,100)
dtst = minibatch(xtst,ytst,100)

println(typeof(dtrn))

function convolution(inputs, num_outputs, kernelsize, mask_type, padding)
kernel_h, kernel_w = kernelsize
height,width,channel,batch = size(inputs)
center_h = convert(Int32,floor(kernel_h / 2))
center_w = convert(Int32,floor(kernel_w / 2))
mask = ones(Float32,(kernel_h, kernel_w, channel, num_outputs))
for j in center_w+1:kernel_w; mask[center_h,j, : , : ]=zeros(size(mask[center_h,j, : , : ]));end
for i in center_h+1:kernel_h; mask[i,: ,:,:]=zeros(size(mask[i,: ,:,:]));end
if mask_type == 'a'
    mask[center_h,center_w,:,:] = zeros(size(mask[center_h,center_w,:,:]))
end
mask .*= randn(kernel_h, kernel_w)
outputs = conv4(mask,inputs, stride=1, padding=padding)
return outputs
end

conv_i = []
for (x,y) in dtrn
conv_inputs=convolution(x, 16, [7 7], 'a',(3,3))
push!(conv_i,conv_inputs)
end
println(typeof(conv_i))

println(size(conv_i))

last_hid =conv_i
c=0
for idx in 1:7
    last_hid = last_hid
    for x in last_hid
        last_hid = (convolution(x, 3, [3 3], 'b',(1,1)))
        println(c,size(last_hid))
    end
end

last_hid =last_hid
for idx in 1:7
    last_hid = last_hid
    for x in last_hid
        last_hid = (convolution(x, 32, [1 1], 'b',(1,1)))
        println(size(last_hid))
    end
end
