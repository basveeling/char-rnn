require 'audio'
require 'csvigo'
-- Modified from https://github.com/oxford-cs-ml-2015/practical6
-- the modification included support for train/val/test splits

local SpectroMinibatchLoader = {}
SpectroMinibatchLoader.__index = SpectroMinibatchLoader

function SpectroMinibatchLoader.create(data_dir, batch_size, seq_length, cutoff_low, cutoff_high, split_fractions)
    -- split_fractions is e.g. {0.9, 0.05, 0.05}

    local self = {}
    setmetatable(self, SpectroMinibatchLoader)

    local song_names = {}
    local singer_songwriter = {"ClaraBerryAndWooldog_Stella",
        "MusicDelta_Country1",
        "AClassicEducation_NightOwl",
        "AlexanderRoss_GoodbyeBolero",
        "ClaraBerryAndWooldog_WaltzForMyVictims",
        "HezekiahJones_BorrowedHeart",
        "FamilyBand_Again",
        "ClaraBerryAndWooldog_TheBadGuys",
        "LizNelson_Rainfall",
        "MusicDelta_Beatles",
        "AimeeNorwich_Child",
        "ClaraBerryAndWooldog_Boys",
        "FacesOnFilm_WaitingForGa",
        "AlexanderRoss_VelvetCurtain",
        "ClaraBerryAndWooldog_AirTraffic",
        "LizNelson_Coldwar",
        "PortStWillow_StayEven",
        "InvisibleFamiliars_DisturbingWildlife",
        "CelestialShore_DieForUs" }
    song_names = singer_songwriter
    self.batch_size = batch_size
    self.seq_length = seq_length
    self.batch_song = {}
    local song_start = 4
    local song_stop = 10


    local reset_rnn = {1 }
    for i=song_start,song_stop do
        local song_name = song_names[i]
        -- construct a tensor with all the data
        local song_path = SpectroMinibatchLoader.get_song_path(song_name,data_dir)
        if not path.exists(song_path) then
            print('one-time setup: preprocessing audio file/annotations for ', song_name)
            SpectroMinibatchLoader.audio_to_tensor(song_name, data_dir, cutoff_low, cutoff_high)
        end
    end
    for i=song_start,song_stop do
        local song_name = song_names[i]
        -- construct a tensor with all the data
        song_path = SpectroMinibatchLoader.get_song_path(song_name,data_dir)
        print('Processing', song_name)
        local song_data = torch.load(song_path)
        local spectro = song_data.spectro[{{},{cutoff_low,cutoff_high}}]:contiguous()
        local voicing = song_data.voicing
        local times = song_data.times

        for i=1,spectro:size(2) do
           -- normalize each channel globally:
           local mean_i = spectro[{ {},i}]:mean()
           local std_i = spectro[{ {},i }]:std()
           spectro[{ {},i }]:add(-mean_i)
           spectro[{ {},i }]:div(std_i)
           collectgarbage()
        end

        -- cut off the end so that it divides evenly
        local len = spectro:size(1)
        if len % (batch_size * seq_length) ~= 0 then
            spectro = spectro:sub(1, batch_size * seq_length
                        * math.floor(len / (batch_size * seq_length)))
        end
        local len = spectro:size(1)
        local ydata = voicing[{{1,len}}]:add(1)
        local timedata = times[{{1,len}}]

        self.input_dim = spectro:size(2)
        local song_x_batches = spectro:view(batch_size, -1,self.input_dim):split(seq_length, 2)  -- #rows = #batches
        local song_y_batches = ydata:view(batch_size, -1):split(seq_length, 2)  -- #rows = #batches
        local song_time_batches = timedata:view(batch_size, -1):split(seq_length, 2)
        if self.x_batches == nil then
            self.x_batches = song_x_batches
            self.y_batches = song_y_batches
            self.time_batches = song_time_batches
            for b=1,#self.x_batches do self.batch_song[b] = song_name end
        else
            -- Concatenate batches
            local offset = #self.x_batches
            table.insert(reset_rnn,#self.x_batches+1,1)
            for i,v in ipairs(song_x_batches) do self.x_batches[offset + i] = v end
            for i,v in ipairs(song_y_batches) do self.y_batches[offset + i] = v end
            for i,v in ipairs(song_time_batches) do self.time_batches[offset + i] = v end
            for b=offset,#self.x_batches do self.batch_song[b] = song_name end
        end
        song_data = nil
        collectgarbage()
    end

    self.reset_rnn = reset_rnn
    print('reshaping tensor...')

    self.nbatches = #self.x_batches
    -- assert(#self.x_batches == #self.y_batches)

    self.ntrain = math.floor(self.nbatches * split_fractions[1])
    self.nval = math.floor(self.nbatches * split_fractions[2])
    self.ntest = self.nbatches - self.nval - self.ntrain -- the rest goes to test (to ensure this adds up exactly)
    self.nval = self.nval + self.ntest
    self.ntest = 0
    self.split_sizes = {self.ntrain, self.nval, self.ntest}
    self.batch_ix = {0,0,0}
    print(self.batch_song)
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
    local do_reset = self.reset_rnn[ix] == 1
    -- Now also returning time
    return self.x_batches[ix], self.y_batches[ix], self.time_batches[ix]
end
function SpectroMinibatchLoader.get_song_path(song_name,data_dir)
    return path.join(data_dir,"torch",string.format("%s_data.t7",song_name))
end
-- *** STATIC method ***
function SpectroMinibatchLoader.audio_to_tensor(song_name, data_dir)
    local timer = torch.Timer()

    print('loading audio/annotations...')
    local audio_path =path.join(data_dir,string.format("Audio/%s_MIX.wav",song_name))
    print(audio_path)
    local song_audio = audio.load(audio_path)
    local csv_path = tostring(path.join(data_dir,string.format("Annotations/Melody_Annotations/MELODY2/%s_MELODY2.csv",song_name)))
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
    song_data.spectro = spectro:t()
    song_data.frequencies = frequencies
    song_data.voicing = voicing
    song_data.unvoicing = unvoicing
    song_data.one_hot_voicing = one_hot_voicing
    song_data.times = times
    -- save output preprocessed files
    song_path = SpectroMinibatchLoader.get_song_path(song_name,data_dir)
    print('saving ' .. song_path)

    torch.save(song_path,song_data)
    song_data = nil
    spectro = nil
    collectgarbage()
end

return SpectroMinibatchLoader

