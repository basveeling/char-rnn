
--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on 
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--
--require('mobdebug').start()
require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'
local SpectroMinibatchLoader = require 'util.SpectroMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','/Users/bas/Downloads/MedleyDB_sample/','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 20, 'size of LSTM internal state')
cmd:option('-num_layers', 5, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'for now only lstm is supported. keep fixed')
-- optimization
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.97,'learning rate decay')
cmd:option('-learning_rate_decay_after',10,'in number of epochs, when to start decaying the learning rate')
cmd:option('-decay_rate',0.95,'decay rate for rmsprop')
cmd:option('-dropout',0.5,'dropout to use just before classifier. 0 = no dropout')
cmd:option('-seq_length',40,'number of timesteps to unroll for')
cmd:option('-batch_size',30,'number of sequences to train on in parallel')
cmd:option('-max_epochs',30,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at')
cmd:option('-train_frac',.9,'fraction of data that goes into train set')
cmd:option('-val_frac',.1,'fraction of data that goes into validation set')
            -- note: test_frac will be computed as (1 - train_frac - val_frac)
-- bookkeeping
cmd:option('-seed',1234,'torch manual random number generator seed')
cmd:option('-print_every',5,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',200,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- Audio options
cmd:option('-cutoff_low',20,'lower cutoff the spectrogram')
cmd:option('-cutoff_high',599,'upper cu toff the spectrogram')
cmd:option('-save_activations_layer',0,'1:2*num_layers+1,  0 for disabled. save csv files of final layer neuron activations for use in sonic visualiser. even numbers are gated, uneven are ungated.')
-- GPU/CPU
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - opt.train_frac - opt.val_frac)
local split_sizes = {opt.train_frac, opt.val_frac, test_frac} 

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
-- -- create the data loader class
local loader = SpectroMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, opt.cutoff_low, opt.cutoff_high, split_sizes)
local input_dim = loader.input_dim
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
print('creating an LSTM with ' .. opt.num_layers .. ' layers'..input_dim, opt.rnn_size, opt.dropout)
protos.rnn = LSTM.lstm(input_dim, opt.rnn_size, opt.num_layers, opt.dropout)
-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    table.insert(init_state, h_init:clone())
end
-- training criterion (negative log likelihood)
protos.criterion = nn.ClassNLLCriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn)

-- initialization
params:uniform(-0.08, 0.08) -- small numbers uniform

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    collectgarbage()
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)

    -- This matrix records the current confusion across classes
    local confusion = optim.ConfusionMatrix(2)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}
    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y, time_batch = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1]) }
            -- Save activations further
            if opt.save_activations_layer > 0 then
                for b=0,opt.batch_size-1 do
                    local activations_off = (t+b*opt.seq_length) + (loader.ntrain + loader.batch_ix[2]-1)*opt.batch_size*opt.seq_length
                    for n=1,opt.rnn_size do
                        -- add row to activations table at time activations_off for neuron n
                        activations[n][activations_off] = {time_batch[{b+1, t}],
                            clones.rnn[t].outnode.data.mapindex[opt.save_activations_layer].module.output[{b+1,{n}}][1],
--                            loader.batch_song[loader.ntrain + loader.batch_ix[2]],
                            'eval'
                        }
                    end
                end
            end
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            local prediction = lst[#lst]

            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])

            -- update confusion
            local classProbabilities = torch.exp(prediction)
            local _, classPredictions = torch.max(classProbabilities, 2)
--            print(classPredictions[{{},1}], y[{{}, t}])
            for k=1,opt.batch_size do
                confusion:add(classPredictions[k][1], y[{{}, t}][k])
            end
        end
        -- carry over lstm state
        if do_reset then
            print("\t Resetting initial state for evaluation...")
            rnn_state[0] = clone_list(init_state)
        else
            rnn_state[0] = rnn_state[#rnn_state]
        end
        print(i .. '/' .. n .. '...')
    end

    loss = loss / opt.seq_length / n

    -- print confusion matrix
    print(confusion)
    print("Average loss",loss)
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
reset_count = 1
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y, time_batch = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------
    local rnn_state = {}
    if do_reset then
        print(string.format("Resetting rnn state (new song) (reset_count %d)",reset_count))
        if reset_count > 1 then
            -- Take iterative average over initial state of rnn.
            for i,L_state in ipairs(init_state_global) do
                init_state[i] = init_state[i] + (init_state_global[i] - init_state[i]) * (1/reset_count)
            end
        else
            init_state = init_state_global
        end
        reset_count = reset_count + 1
        rnn_state = {[0] = clone_list(init_state) }
    else
        rnn_state = {[0] = init_state_global }
    end
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        -- for debugging: Save activations of final layer neurons per timestep
        if opt.save_activations_layer > 0 then
            for b=0,opt.batch_size-1 do
                local activations_off = (t+b*opt.seq_length) + (loader.batch_ix[1]-1)*opt.batch_size*opt.seq_length
                for n=1,opt.rnn_size do
                    activations[n][activations_off] = {time_batch[{b+1, t}],
                        clones.rnn[t].outnode.data.mapindex[opt.save_activations_layer].module.output[{b+1,{n}}][1],
--                        loader.batch_song[loader.batch_ix[1]],
                        'train'
                    }
                end
            end
        end
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        predictions[t] = lst[#lst] -- last element is the prediction
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        table.insert(drnn_state[t], doutput_t)
        local dlst = clones.rnn[t]:backward({x[{{}, t}], unpack(rnn_state[t-1])}, drnn_state[t])
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k > 1 then -- k == 1 is gradient on x, which we dont need
                -- note we do k-1 because first item is dembeddings, and then follow the 
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)

    ----------------- graphhhh -------------------------

--    graph.dot(clones.rnn[1].fg, 'Forward Graph','/tmp/fg')
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}
local optim_state = {learningRate = opt.learning_rate, alpha = opt.decay_rate}
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
collectgarbage()

-- Initialize activation logging
if opt.save_activations_layer then
    activations = {}
    for n=1, opt.rnn_size do activations[n] = {} end
end
for i = 1, iterations do
    local epoch = i / loader.ntrain

    local timer = torch.Timer()
    local _, loss = optim.rmsprop(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = loss[1] -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % loader.ntrain == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
            print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = eval_split(2) -- 2 = validation
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader.vocab_mapping
        torch.save(savefile, checkpoint)
        -- Output final layer neuron activations
        if opt.save_activations_layer > 0 then
            for n=1,opt.rnn_size do
                -- sort activations
                local function compare_activation(a,b)
                  return a[1] < b[1]
                end
                table.sort(activations[n],compare_activation)
                -- save to csv in /tmp
                csvigo.save({verbose=false,path=string.format("/tmp/activations_L%d_neuron-%d_iteration-%d.csv",opt.save_activations_layer,n,i),data=activations[n],headers=false})
            end
        end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), train_loss = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end
   
    if i % 1 == 0 then collectgarbage() end -- TODO: fix memory issues

    -- handle early stopping if things are going really bad
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


