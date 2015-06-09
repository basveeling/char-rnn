
# melody-rnn
This is an adaption of [karpathy/char-rnn](/karpathy/char-rnn) for melody onset/offset detection. 

## Method
- A spectrogram is generated of each song, a subset of the melodic range is selected
- A melody on/off vector is build based on melody annotations from the dataset
- The RNN is trained similarly to char-rnn 

## Dataset
The [MedleyDB](http://medleydb.weebly.com/) dataset is used. A sample of 2 songs can be downloaded [here](http://marl.smusic.nyu.edu/medleydb_webfiles/MedleyDB_sample.tar.gz).

## License
MIT
