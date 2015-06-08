require 'audio'
require 'csvigo'
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local SpectroMinibatchLoader = {}
SpectroMinibatchLoader.__index = SpectroMinibatchLoader

function SpectroMinibatchLoader.create(data_dir, batch_size, seq_length, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, SpectroMinibatchLoader)

    local song_name = "Phoenix_ScotchMorris" --"LizNelson_Rainfall"
    local data_file = path.join(data_dir, song_name..'.t7')

    -- construct a tensor with all the data
    if not path.exists(data_file) then
        print('one-time setup: preprocessing audio file/annotations ' .. song_name .. '...')
        SpectroMinibatchLoader.audio_to_tensor(song_name, data_file)
    end

    print('loading data files...')
    local data = torch.load(data_file)
    local t_spectro = data.spectro:t()
    local frequencies = data.frequencies
    local voicing = data.voicing
    local unvoicing = data.unvoicing
    local one_hot_voicing = data.one_hot_voicing
    local times = data.times
    t_spectro = t_spectro[{{},{100,800-1}}]:contiguous() -- Crop spectrogram
    mean = {}
    std = {}
    for i=1,t_spectro:size(2) do
       -- normalize each channel globally:
       mean[i] = t_spectro[{ {},i}]:mean()
       std[i] = t_spectro[{ {},i }]:std()
       t_spectro[{ {},i }]:add(-mean[i])
       t_spectro[{ {},i }]:div(std[i])
    end
    -- cut off the end so that it divides evenly
    local len = t_spectro:size(1)

    if len % (batch_size * seq_length) ~= 0 then
        print('cutting off end of data so that the batches/sequences divide evenly')
        t_spectro = t_spectro:sub(1, batch_size * seq_length 
                    * math.floor(len / (batch_size * seq_length)))
    end
    local len = t_spectro:size(1)
    self.input_dim = t_spectro:size(2)
    print(self.input_dim)
    self.len = len
    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = voicing[{{1,len}}]:add(1)
    -- TODO: add some kind of normalizing?
    self.x_batches = t_spectro:view(batch_size, -1,self.input_dim):split(seq_length, 2)  -- #rows = #batches
    self.nbatches = #self.x_batches
    self.y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
    -- assert(#self.x_batches == #self.y_batches)

    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)

    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}

    print(string.format('data load done. Number of batches in train: %d, val: %d, test: %d', self.ntrain, self.nval, self.ntest))
    collectgarbage()
    return self
end

function SpectroMinibatchLoader:reset_batch_pointer(split_index, batch_index)
    batch_index = batch_index or 0
    self.batch_ix[split_index] = batch_index
end
 
function SpectroMinibatchLoader:next_batch(split_index)
    -- split_index is integer: 1 = train, 2 = val, 3 = test
    self.batch_ix[split_index] = self.batch_ix[split_index] + 1
    if self.batch_ix[split_index] > self.split_sizes[split_index] then
        self.batch_ix[split_index] = 1 -- cycle around to beginning
    end
    -- pull out the correct next batch
    local ix = self.batch_ix[split_index]
    if split_index == 2 then ix = ix + self.ntrain end -- offset by train set size
    if split_index == 3 then ix = ix + self.ntrain + self.nval end -- offset by train + test
    return self.x_batches[ix], self.y_batches[ix]
end

-- *** STATIC method ***
function SpectroMinibatchLoader.audio_to_tensor(song_name,data_file)
    local timer = torch.Timer()

    print('loading audio/annotations...')
    song_audio = audio.load("/Users/bas/Downloads/MedleyDB_sample/Audio/"..song_name.."/"..song_name.."_MIX.wav")
    melody = csvigo.load({path="/Users/bas/Downloads/MedleyDB_sample/Annotations/Melody_Annotations/MELODY2/"..song_name.."_MELODY2.csv",header=false})

    print("Spectrogramify this song...")
    -- A stride of 256 corresponds with the annotation stride of MedleyDB, the windowsize determines the frequency precision and inversely the duration precisino
    spectro = audio.spectrogram(song_audio[{{1},{}}], 2^12, 'hann', 256)
    print("Done!")

    times = torch.Tensor(melody.var_1)
    frequencies = torch.Tensor(melody.var_2)
    voicing = frequencies:gt(0):double()
    unvoicing = torch.Tensor(voicing:size()):fill(1) - voicing
    one_hot_voicing = torch.cat(unvoicing,voicing, 2)

    -- save output preprocessed files
    print('saving ' .. data_file)
    data = {}
    data.spectro = spectro
    data.frequencies = frequencies
    data.voicing = voicing
    data.unvoicing = unvoicing
    data.one_hot_voicing = one_hot_voicing
    data.times = times
    torch.save(data_file,data)
end

return SpectroMinibatchLoader

