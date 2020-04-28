using Pkg; installed = Pkg.installed()
for p in ("Knet","KnetLayers","Images","Statistics","Plots","IterTools","ImageMagick")
    haskey(installed,p) || Pkg.add(p)
end
using Base.Iterators: flatten
using IterTools: ncycle, takenth
using Statistics: mean
using KnetLayers, Images, Plots, Statistics, ImageMagick
using Knet: Knet, conv4, pool, mat, KnetArray, nll, zeroone, progress, sgd, param, param0, dropout, relu, Data, sigmoid, sigm, minibatch,Random,xavier_uniform,xavier_normal
include(Knet.dir("data","mnist.jl"))
#Pkg.update("Knet")
#Pkg.build("Knet")

xtrn,ytrn,xtst,ytst, = mnist()
println.(summary.((xtrn,ytrn,xtst,ytst)));
#module list
#nvidia-smi

function binarize(images)
Random.seed!(123)
    ran = rand(Float32,size(images))
    for i=1:size(images,1)
        for j=1:size(images,2)
            for k=1:size(images,3)
                for l=1:size(images,4)
                    if images[i,j,k,l]<=ran[i,j,k,l]
                       images[i,j,k,l] = 0
                    else
                        images[i,j,k,l] = 1
                    end
                end
            end
        end
    end
    return images
end

xtrn = binarize(xtrn)
xtst = binarize(xtst)
dtrn = minibatch(xtrn,xtrn,100;xtype=KnetArray,ytype=KnetArray)
dtst = minibatch(xtst,xtst,100;xtype=KnetArray,ytype=KnetArray)

println(typeof(dtrn))

(x,y) = first(dtrn)
println.(summary.((x,y)))

function mask(inputs, num_outputs, kernelsize, mask_type)
kernel_h, kernel_w = kernelsize, kernelsize
height,width,channel,batch = inputs
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

function minit(m,mask_type)
kernel_h, kernel_w, channel, num_outputs = size(m)
center_h = convert(Int32,(kernel_h+1) / 2)
center_w = convert(Int32,(kernel_w+1) / 2)
#init = ones(Float32,(kernel_h, kernel_w, channel, num_outputs))
for j in center_w+1:kernel_w; m[center_h,j, : , : ]=zeros(size(m[center_h,j, : , : ]));end
for i in center_h+1:kernel_h; m[i,: ,:,:]=zeros(size(m[i,: ,:,:]));end
if mask_type == 'a'
    m[center_h,center_w,:,:] = zeros(size(m[center_h,center_w,:,:]))
end
return m
end

function displayimage(image)
#display images
result = Any[]
K = Any[]
for i=1:10
    result_col = Any[]
    for j=1:10
        k=(i-1)*10+j
        push!(result_col,image[:, :,1,k])
    end
    push!(result,hcat(result_col...))
end
return result
#colorview(Gray, vcat(r8...))
end

function softloss(ypred, ygold)
#    println(1,maximum(Array(ypred)),minimum(Array(ypred)))
#    println(2,maximum(Array(ygold)),minimum(Array(ygold)))
#    ynorm = Array(ypred) .- log.(sum(exp.(Array(ypred))))
#    println(3,maximum(ynorm),minimum(ynorm))
#    los = -sum(Array(ygold) .* ynorm)/(100*28*28)
a = max.(zeros(Float32,size(Array(ypred))),Array(ypred))
#println('a',mean(a))
b = Array(ypred) .* Array(ygold)
#println('b',mean(b))
c = log.(1 .+ exp.(-abs.(Array(ypred))))
#println('c',mean(c))
los = a .- b .+ c
#     los = maxvalue(zeros(Float32,size(Array(ypred))),Array(ypred)) .- Array(ypred) .* Array(ygold) + log.(1 .+ exp.(-abs.(Array(ypred))))
#    println(4,maximum(los),minimum(los))
l=mean(los)
#println(l)
#    return los
     return l
end

function rmspropupdate(w,g,G)
lr::Float32    = 0.001
rho::Float32   = 0.9
eps::Float32   = 1e-6
gclip::Float32 = 1.0
g = min.(gclip .*ones(Float32,size(Array(g))),Array(g))
g = max.(.-gclip .*ones(Float32,size(Array(g))),Array(g))
G  .= Array(G) .*rho + Array(g).^2 .*(1-rho)
w  .= KnetArray(Array(w) - Array(g) .*lr ./ sqrt.(G .+ eps)) 
return w,G
end

#op=SGD(lr=0.001)
#Rmsprop(lr=0.001, gclip=0, rho=0.9, eps=1e-6)
#m1 = Param(mask((28,28,1,100), 16, 7, 'a'));
#m2 = Param(mask((28,28,16,100), 3, 3, 'b'));
#m31 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m32 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m33 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m34 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m35 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m36 = Param(mask((28,28,3,100), 3, 3, 'b'));
#m4 = Param(mask((28,28,3,100), 32, 1, 'b'));
#m5 = Param(mask((28,28,32,100), 32, 1, 'b'));
#m6 = Param(mask((28,28,32,100), 1, 1, 'b'));


m1=Param(KnetArray(minit(xavier_uniform(Float32,7,7,1,16;gain=1),'a')))
m2=Param(KnetArray(minit(xavier_uniform(Float32,3,3,16,3;gain=1),'b')))
m31=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m32=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m33=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m34=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m35=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m36=Param(KnetArray(minit(xavier_uniform(Float32,3,3,3,3;gain=1),'b')))
m4=Param(KnetArray(minit(xavier_uniform(Float32,1,1,3,32;gain=1),'b')))
m5=Param(KnetArray(minit(xavier_uniform(Float32,1,1,32,32;gain=1),'b')))
m6=Param(KnetArray(minit(xavier_uniform(Float32,1,1,32,1;gain=1),'b')))


function main(data,m10,m20,m310,m320,m330,m340,m350,m360,m40,m50,m60) #,m320,m330,m340,m350,m360,m40
#conv_i = Knet.batchnorm(conv4(m10,data, stride=1, padding=(3,3));training=true)
conv_i = conv4(m10,data, stride=1, padding=(3,3))

#first loop 7 times
#outputs1 = Knet.batchnorm(conv4(m20,conv_i, stride=1, padding=(1,1));training=true)
outputs1 = conv4(m20,conv_i, stride=1, padding=(1,1))

#outputs2 = Knet.batchnorm(conv4(m310,outputs1, stride=1, padding=(1,1));training=true)
outputs2 = conv4(m310,outputs1, stride=1, padding=(1,1))
#outputs3 = Knet.batchnorm(conv4(m320,outputs2, stride=1, padding=(1,1));training=true)
outputs3 = conv4(m320,outputs2, stride=1, padding=(1,1))
#outputs4 = Knet.batchnorm(conv4(m330,outputs3, stride=1, padding=(1,1));training=true)
outputs4 = conv4(m330,outputs3, stride=1, padding=(1,1))
#outputs5 = Knet.batchnorm(conv4(m340,outputs4, stride=1, padding=(1,1));training=true)
outputs5 = conv4(m340,outputs4, stride=1, padding=(1,1))
#outputs6 = Knet.batchnorm(conv4(m350,outputs5, stride=1, padding=(1,1));training=true)
outputs6 = conv4(m350,outputs5, stride=1, padding=(1,1))
#outputs7 = Knet.batchnorm(conv4(m360,outputs6, stride=1, padding=(1,1));training=true)
outputs7 = conv4(m360,outputs6, stride=1, padding=(1,1))

#second loop relu 2 times
#temp8 = Knet.batchnorm(conv4(m40,outputs7, stride=1, padding=(0,0));training=true)
temp8 = conv4(m40,outputs7, stride=1, padding=(0,0))
#outputs8 = KnetArray(relu.(Array(temp8))./maximum(Array(temp8)))
outputs8 = KnetArray(relu.(Array(temp8)))

#temp9 = Knet.batchnorm(conv4(m50,outputs8, stride=1, padding=(0,0));training=true)
temp9 = conv4(m50,outputs8, stride=1, padding=(0,0))
outputs9 = KnetArray(relu.(Array(temp9)))
#outputs9 = KnetArray(relu.(Array(temp9))./maximum(Array(temp9)))

#output sigmoid
#conv2d_out_logits = Knet.batchnorm(conv4(m60,outputs9, stride=1, padding=(0,0));training=true)
conv2d_out_logits = conv4(m60,outputs9, stride=1, padding=(0,0))
#output = sigmoid.(Array(conv2d_out_logits))
return conv2d_out_logits
end     #end of main function

G1  = zeros(Float32,size(m1))
G2  = zeros(Float32,size(m2))
G31 = zeros(Float32,size(m31))
G32 = zeros(Float32,size(m32))
G33 = zeros(Float32,size(m33))
G34 = zeros(Float32,size(m34))
G35 = zeros(Float32,size(m35))
G36 = zeros(Float32,size(m36))
G4  = zeros(Float32,size(m4))
G5  = zeros(Float32,size(m5))
G6  = zeros(Float32,size(m6))

trainloss = Any[]
testloss  = Any[]
for epoch in 1:20

#training
epochloss=0
num=0
for (x,y) in dtrn
loss =@diff softloss(main(x,m1,m2,m31,m32,m33,m34,m35,m36,m4,m5,m6),x) #,m32,m33,m34,m35,m36,m4
epochloss=value(loss)+epochloss
num=num+1

lgm1 = grad(loss,m1)
#update!(m1,lgm1,op)
#update!(m1,lgm1)
global m1,G1 = rmspropupdate(m1,lgm1,G1)
global m1=minit(m1,'a')

lgm2 = grad(loss,m2)
#update!(m2,lgm2)
global m2,G2 = rmspropupdate(m2,lgm2,G2)
global m2=minit(m2,'b')

lgm31 = grad(loss,m31)
#update!(m31,lgm31)
global m31,G31 = rmspropupdate(m31,lgm31,G31)
global m31=minit(m31,'b')

lgm32 = grad(loss,m32)
#update!(m32,lgm32)
global m32,G32 = rmspropupdate(m32,lgm32,G32)
global m32=minit(m32,'b')

lgm33 = grad(loss,m33)
#update!(m33,lgm33)
global m33,G33 = rmspropupdate(m33,lgm33,G33)
global m33=minit(m33,'b')

lgm34 = grad(loss,m34)
#update!(m34,lgm34)
global m34,G34 = rmspropupdate(m34,lgm34,G34)
global m34=minit(m34,'b')

lgm35 = grad(loss,m35)
#update!(m35,lgm35)
global m35,G35 = rmspropupdate(m35,lgm35,G35)
global m35=minit(m35,'b')

lgm36 = grad(loss,m36)
#update!(m36,lgm36)
global m36,G36 = rmspropupdate(m36,lgm36,G36)
global m36=minit(m36,'b')

lgm4 = grad(loss,m4)
#update!(m4,lgm4)
global m4,G4 = rmspropupdate(m4,lgm4,G4)
global m4=minit(m4,'b')

lgm5 = grad(loss,m5)
#update!(m5,lgm5,op)
global m5,G5 = rmspropupdate(m5,lgm5,G5)
global m5=minit(m5,'b')

lgm6 = grad(loss,m6)
#update!(m6,lgm6,op)
#update!(m6,lgm6)
global m6,G6 = rmspropupdate(m6,lgm6,G6)
global m6=minit(m6,'b')

end  #end of one dtrn training
trnloss=epochloss/num
println("trnloss ",trnloss)
push!(trainloss,trnloss)

#testing
epochloss=0
num=0
for (x,y) in dtst
loss =softloss(main(x,m1,m2,m31,m32,m33,m34,m35,m36,m4,m5,m6),x)   #,m32,m33,m34,m35,m36,m4
epochloss=loss+epochloss
num=num+1
end  #end of one dtst testing
tstloss=epochloss/num
println("tstloss ",tstloss)
push!(testloss,tstloss)


#generate images
d, r = divrem(epoch, 4)
if r==0
(x,y)=first(dtst)
samples = KnetArray(zeros(Float32,(28, 28, 1,100)))
samples[1:14,:,1,:] = x[1:14,:,1,:]
for i=14:28
    for j=1:28
        next_sample = KnetArray(binarize(sigmoid.(Array(main(samples,m1,m2,m31,m32,m33,m34,m35,m36,m4,m5,m6)))))   #,m32,m33,m34,m35,m36,m4
        samples[i, j,1,:] = next_sample[i, j,1,:]
    end
end

#display and save result
if d==1
R = displayimage(samples);
Knet.@save abspath(joinpath("SAVEfulllayer","resultM_4epoch.jld2")) R
elseif d==2
R = displayimage(samples);
Knet.@save abspath(joinpath("SAVEfulllayer","resultM_8epoch.jld2")) R
elseif d==3
R = displayimage(samples);
Knet.@save abspath(joinpath("SAVEfulllayer","resultM_12epoch.jld2")) R
elseif d==4
R = displayimage(samples);
Knet.@save abspath(joinpath("SAVEfulllayer","resultM_16epoch.jld2")) R
elseif d==5
R = displayimage(samples);
Knet.@save abspath(joinpath("SAVEfulllayer","resultM_20epoch.jld2")) R

#nl = nll(samples,x)
#push!(nllvalue,nl)
#println(nl)
end  #end display and save result

end  #end generate images

end  #end of epoch

Knet.@save abspath(joinpath("SAVEfulllayer","trnlossM_20epoch.jld2")) trainloss
Knet.@save abspath(joinpath("SAVEfulllayer","tstlossM_20epoch.jld2")) testloss
#Knet.@save abspath(joinpath("SAVEfulllayer","nllvalueM_20epoch.jld2")) nllvalue