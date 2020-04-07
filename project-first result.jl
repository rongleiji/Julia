# Set-Up related files and Hyper-parameters
using Pkg; for p in ["Knet","ArgParse"]; haskey(Pkg.installed(),p) || Pkg.add(p); end
using Knet
import Knet: train!
using Printf, Dates, Random
STDOUT = Base.stdout
using ArgParse
import Base: length
include(Pkg.dir("Knet","data","wikiner.jl"));

@doc make_instance

# make_instances procedure is given to you
function make_instances(data, w2i, t2i)
    words = []; tags = []
    for k = 1:length(data)
        this_words, this_tags = make_instance(data[k], w2i, t2i)
        push!(words, this_words)
        push!(tags, this_tags)
    end
    order = sortperm(words, by=length, rev=true)
    return words, tags
end

#=
You need to implement make_instance function
instance is a list of tuples. Each tuple contains a word and the corresponding tag as string.
You need to convert them into indices using word to index (w2i) and tag to index (t2i)
=#
function make_instance(instance, w2i, t2i)
    input = Int[]
    output = Int[] 
    # START ANSWER
    input = map(i->get(w2i, instance[i][1], w2i[UNK]), [1:length(sample)...])
    input = reshape(input, 1, length(input))
    input = convert(Array{Int64}, input)
        
    output = map(i->t2i[instance[i][1]], [1:length(sample)...])
    output = reshape(output, 1, length(output))
    output = convert(Array{Int64}, output)
    # END ANSWER
    return input, output
end

#=
This struct contains processed data (e.g words and tags are indices)
and necessary variables to prepare minibatches.
WikiNERProcessed struct works as an iterator.
=#
mutable struct WikiNERProcessed
    words
    tags
    batchsize
    ninstances
    shuffled
end


function WikiNERProcessed(instances, w2i, t2i; batchsize=16, shuffled=true)
    words, tags = make_instances(instances, w2i, t2i)
    ninstances = length(words)
    return WikiNERProcessed(words, tags, batchsize, ninstances, shuffled)
end


function length(d::WikiNERProcessed)
    d, r = divrem(d.ninstances, d.batchsize)
    return r == 0 ? d : d+1
end

#=
You will use the RNN callable object in your model. It supports variable length instances in its input.
However, you need to prepare your input such as the RNN object can work on it. See the batchSizes option of the RNN object.
=#
function iterate(d::WikiNERProcessed, state=ifelse(d.shuffled, randperm(d.instances), 1:d.ninstances))
    # START ANSWER
    n = length(state)
    n == 0 && return nothing
    batchsize = min(d.batchsize, n)
    indices, new_state = state[1:batchsize], state[batchsize+1:end]
    words, tags = d.words[:, indices], d.tags[:, indices]
    # END ANSWER
    return ((words, tags, batchsizes), new_state)
end

@doc rnnforw

# DO NOT TOUCH CELL, take advantage of _usegpu and _atype in the following parts
_usegpu = gpu()>=0
_atype = ifelse(_usegpu, KnetArray{Float32}, Array{Float64})


mutable struct Embedding
    w # weight
end


mutable struct Linear
    w # weight
    b # bias
end


mutable struct Hidden
    w # weight
    b # bias
    fun # non-linear activation function like relu or tanh
end

@doc param

# initializations
function Embedding(vocabsize::Int, embedsize::Int, atype=_atype, scale=0.01)
    w = Param(convert(atype, scale*randn(embedsize, vocabsize)));
    return Embedding(w)
end


function Linear(xsize::Int, ysize::Int, atype=_atype, scale=0.01)
    # start your answer
    w = Param(convert(atype, scale*randn(ysize,xsize)));
    b = Param(convert(atype, scale*randn(ysize)));
    return Linear(w,b)
    # end your answer
end


function Hidden(xsize::Int, ysize::Int, fun=relu, atype=_atype, scale=0.1)
    # start your answer
    w = Param(convert(atype, scale*randn(ysize,xsize)));
    b = Param(convert(atype, scale*randn(ysize)));
    return Hidden(w,b,fun)
    # end your answer
end

# forward propagations
function (l::Embedding)(x)
    l.w[:, x]
end


function (l::Linear)(x)
    # start your answer
   l.w * mat(x,dims=1) .+ l.b 
    # end your answer
end


function (l::Hidden)(x)
    # start your answer
    l.fun(l.w * mat(x,dims=1) .+ l.b)
    # end your answer
end

# DO NOT TOUCH THIS CELL
mutable struct NERTagger
    embed_layer::Embedding
    rnn::RNN
    hidden_layer::Hidden
    softmax_layer::Linear
end

# model initialization
# Check the array type (cpu vs gpu)
# Initialize your modules using given arguments
function NERTagger(rnn_hidden, words, tags, embed, mlp_hidden, usegpu=_usegpu, winit=0.01)
    # start your answer
    w = Array{Any}(6)
    input = embed
    srnn, wrnn = rnninit(input, rnn_hidden; bidirectional=true, usegpu=_usegpu)
    w[1] = wrnn
    w[2] = convert(atype, winit*randn(mlp_hidden, 2*rnn_hidden))
    w[3] = convert(atype, zeros(mlp_hidden, 1))
    w[4] = convert(atype, winit*randn(tags, mlp_hidden))
    w[5] = convert(atype, winit*randn(tags, 1))
    w[6] = convert(atype, winit*randn(embed, words))
    return w, srnn
    # end your answer
    end 

# model forward propagation
# Call your modules as described in the introduction section
function (m::NERTagger)(x, batchsizes)
    # start your answer
    cwords,cinds,rwods,rinds,bs = x
    cembed = w[end-1][:,cwords]
    cembed = reshape(cembed, size(cembed,1),size(cembed)[end])
    length(rinds)==0 && return cembed
    r = srnn; wr = w[2]
    c0 = w[end][:,rwords]
    y, hy,cy = rnnforw(r,wr,c0;hy=true,cy=true,batchSizes=bs)
    r0 = permutedims(hy, (3,1,2))
    rembed = reshape(r0, size(r0,1)*size(r0,2),size(r0,3))
    e0 = hcat(cembed,rembed)
    e1 = e0[:,[cinds...,rinds...]]
    return e1
    # end your answer
end

# Get your probabilities from your model
# Calculate the loss function for average per token.
function (m::NERTagger)(x, batchsizes, ygold)
    # start your answer
    ws = m.w; xs = x
    wx = ws[6]
    r = srnn; wr = ws[1]
    wnlp = ws[2];bmlp = ws[3]
    wy = ws[4]; by = ws[5]
    x = wx[:, xs]
    y, hy, cy = rnnforw(r, wr, x)
    y2 = reshape(y, size(y,1),size(y,2)*size(y,3))
    yw = wmlp*y2.+bmlp
    return nll(wy*y3.+by, ygold)
    # end your answer
end

# TODO: with iterators

# possible helpful procedures: argmax, vec
function accuracy(m::NERTagger, data, i2t)
    ncorrect = 0
    ntokens = 0
    for (x, ygold, batchsizes) in data
        scores = m(x, batchsizes)
        ntokens= ntokens+1
        # START ANSWER
        if mean(argmax(scores) .== argmax(vec(ygold)))
                ncorrect = ncorrect+1
        end
        # END ANSWER
    end

    return ncorrect/ntokens
end

# DON'T TOUCH this cell
function main(args)
    o = parse_options(args)
    atype = o[:atype]
    display(o)
    o[:seed] > 0 && Knet.seed!(o[:seed])

    # load WikiNER data
    data = WikiNERData()

    # build model
    nwords, ntags = length(data.w2i), data.ntags
    model = NERTagger(o[:hidden], nwords, ntags, o[:embed], o[:mlp])
    initopt!(model)
    # opt = optimizers(w, Adam)

    # make instances
    trn = WikiNERProcessed(data.trn, data.w2i, data.t2i; batchsize=o[:batchsize])
    dev = WikiNERProcessed(data.dev, data.w2i, data.t2i; batchsize=o[:batchsize])

    # train bilstm tagger
    nwords = data.nwords
    ninstances = length(trn)
    println("nwords=$nwords, ntags=$ntags, ninstances=$ninstances"); flush(STDOUT)
    println("startup time: ", Int((now()-t00).value)*0.001); flush(STDOUT)
    t0 = now()
    all_time = dev_time = all_tagged = this_tagged = this_loss = 0
    iter = 0
    while true
        # training
        for (k, (x, ygold, batchsizes)) in enumerate(trn)
            num_tokens = length(x)
            # instance_loss = adam!(model, (x, ygold, batchsizes))
            instance_loss = mytrain!(model, x, batchsizes, ygold)
            this_loss += num_tokens*instance_loss
            this_tagged += num_tokens
            iter += 1
            if iter % o[:report] == 0
                println(this_loss/this_tagged); flush(STDOUT)
            end
            if iter % o[:valid] == 0
                # validation
                dev_start = now()
                tag_acc = accuracy(model, dev, data.i2t)
                dev_time += Int((now()-dev_start).value)*0.001
                train_time = Int((now()-t0).value)*0.001-dev_time

                # report
                @printf("%d iters finished, loss=%f\n", iter, this_loss/this_tagged)
                all_tagged += this_tagged
                this_loss = this_tagged = 0
                all_time = Int((now()-t0).value)*0.001
                @printf("tag_acc=%.4f, time=%.4f, word_per_sec=%.4f\n",
                    tag_acc, train_time, all_tagged/train_time)
                flush(STDOUT)
            end
            iter >= o[:iters] && return
        end

    end
end

function parse_options(args)
    s = ArgParseSettings()
    s.description = "LSTM Tagger in Knet"

    @add_arg_table s begin
        ("--usegpu"; action=:store_true; help="use GPU or not")
        ("--embed"; arg_type=Int; default=128; help="word embedding size")
        ("--hidden"; arg_type=Int; default=50; help="LSTM hidden size")
        ("--mlp"; arg_type=Int; default=32; help="MLP size")
        ("--epochs"; arg_type=Int; default=3; help="number of training epochs")
        ("--iters"; arg_type=Int; default=20000; help="number of training iterations")
        ("--report"; arg_type=Int; default=500; help="report period in iters")
        ("--valid"; arg_type=Int; default=5000; help="valid period in iters")
        ("--seed"; arg_type=Int; default=-1; help="random seed")
        ("--batchsize"; arg_type=Int; default=16; help="batchsize")
    end

    isa(args, AbstractString) && (args=split(args))
    o = parse_args(args, s; as_symbols=true)
    o[:atype] = (gpu() >= 0 && o[:usegpu]) ? KnetArray{Float32} : Array{Float64}
    println(o); flush(STDOUT)
    return o
end


function mytrain!(model::NERTagger, x, batchsizes, ygold)
    values = []
    J = @diff model(x, batchsizes, ygold)
    for par in params(model)
        g = grad(J, par)
        update!(value(par), g, par.opt)
    end
    return value(J)
end

function initopt!(model::NERTagger, optimizer="Adam()")
    for par in params(model)
        par.opt = eval(Meta.parse(optimizer))
    end
end

t00 = now();main("--seed 1 --iters 10000 --usegpu")


