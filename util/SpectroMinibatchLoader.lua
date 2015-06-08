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

    local song_names = {"LizNelson_Rainfall","Phoenix_ScotchMorris"} -- "Phoenix_ScotchMorris" --"LizNelson_Rainfall"
    local data_file = path.join(data_dir, 'all'..'.t7')

    -- construct a tensor with all the data
    if not path.exists(data_file) then
        print('one-time setup: preprocessing audio file/annotations ' .. song_name .. '...')
        SpectroMinibatchLoader.audio_to_tensor(song_names,data_file, data_dir)
    end

    print('loading data files...')
    local all_data = torch.load(data_file)
    for song_data in all_data do

        local spectro = song_data.spectro:t()
        local voicing = song_data.voicing
        spectro = spectro[{{},{100,800-1}}]:contiguous() -- Crop spectrogram
        mean = {}
        std = {}
        for i=1,spectro:size(2) do
           -- normalize each channel globally:
           mean[i] = spectro[{ {},i}]:mean()
           std[i] = spectro[{ {},i }]:std()
           spectro[{ {},i }]:add(-mean[i])
           spectro[{ {},i }]:div(std[i])
        end
        -- cut off the end so that it divides evenly
        local len = spectro:size(1)

        if len % (batch_size * seq_length) ~= 0 then
            print('cutting off end of data so that the batches/sequences divide evenly')
            spectro = spectro:sub(1, batch_size * seq_length
                        * math.floor(len / (batch_size * seq_length)))
        end
        local len = spectro:size(1)
        -- TODO: NEXT UP: propogate changes for having multiple songs
    end
    self.input_dim = all_data[1].spectro:size(2)
    print(self.input_dim)
    self.len = len
    -- self.batches is a table of tensors
    print('reshaping tensor...')
    self.batch_size = batch_size
    self.seq_length = seq_length

    local ydata = voicing[{{1,len}}]:add(1)
    self.x_batches = spectro:view(batch_size, -1,self.input_dim):split(seq_length, 2)  -- #rows = #batches
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
function SpectroMinibatchLoader.audio_to_tensor(song_names,data_file, data_dir)
    local timer = torch.Timer()

    data = {}
    for song_name in song_names do
        print('loading audio/annotations...')
        local audio_path =path.join(data_dir,"Audio/"..song_name.."/"..song_name.."_MIX.wav")
        print(audio_path)
        local song_audio = audio.load(audio_path)
        local csv_path = tostring(path.join(data_dir,"Annotations/Melody_Annotations/MELODY2/"..song_name.."_MELODY2.csv"))
        print(csv_path)
        local melody = csvigo.load({path=csv_path,header=false})

        print("Spectrogramify this song...")
        -- A stride of 256 corresponds with the annotation stride of MedleyDB, the windowsize determines the frequency precision and inversely the duration precisino
        local spectro = audio.spectrogram(song_audio[{{1},{}}], 2^12, 'hann', 256)
        print("Done!")

        local times = torch.Tensor(melody.var_1)
        local frequencies = torch.Tensor(melody.var_2)
        local voicing = frequencies:gt(0):double()
        local unvoicing = torch.Tensor(voicing:size()):fill(1) - voicing
        local one_hot_voicing = torch.cat(unvoicing,voicing, 2)
        local song_data = {}
        song_data.spectro = spectro
        song_data.frequencies = frequencies
        song_data.voicing = voicing
        song_data.unvoicing = unvoicing
        song_data.one_hot_voicing = one_hot_voicing
        song_data.times = times
        data[#data + 1] = song_data
    end
    -- save output preprocessed files
    print('saving ' .. data_file)

    torch.save(data_file,data)
end

return SpectroMinibatchLoader

