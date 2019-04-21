from __future__ import division
from __future__ import print_function
import json
import multiprocessing as mp
import numpy as np
import os
import random
import librosa
import numpy as np
from pydub import AudioSegment
import io

# load from m4a, the format of voxcele2 dataset
def load_m4a(vid_path, sr):
    audio = AudioSegment.from_file(vid_path, "mp4")
    audio = audio.set_frame_rate(sr)
    audio = audio.set_channels(1)
    audio = audio.set_sample_width(2)
    buf = io.BytesIO()
    audio.export(buf, format='s16le')
    wav = np.frombuffer(buf.getbuffer(), np.int16)
    wav = np.array(wav/32768.0, dtype=np.float32)

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output

# load from wav
def load_wav(vid_path, sr):
    wav, sr_ret = librosa.load(vid_path, sr=sr)
    assert sr_ret == sr

    intervals = librosa.effects.split(wav, top_db=20)
    wav_output = []
    for sliced in intervals:
      wav_output.extend(wav[sliced[0]:sliced[1]])
    wav_output = np.array(wav_output)
    return wav_output

def lin_spectogram_from_wav(wav, hop_length, win_length, n_fft=1024):
    linear = librosa.stft(wav, n_fft=n_fft, win_length=win_length, hop_length=hop_length) # linear spectrogram
    return linear.T

def load_data(path, sr=16000, win_length=400, hop_length=160, n_fft=512, rand_duration=2500, is_training=True):
    try:
        if(path.endswith('.wav')):
            wav = load_wav(path, sr=sr)
        elif(path.endswith('.m4a')):
            wav = load_m4a(path, sr=sr)
        else:
            print("!!! Not supported audio format.")
            return None
    except Exception as e:
        print("Exception happened when load_data('{}'): {}".format(path, str(e)))
        return None

    linear_spect = lin_spectogram_from_wav(wav, hop_length, win_length, n_fft)
    mag, _ = librosa.magphase(linear_spect)  # magnitude
    mag_T = mag.T
    freq, time = mag_T.shape
    spec_mag = mag_T

    if(is_training):
        randSpec = rand_duration//(1000//(sr//hop_length))  # random duration in spectrum
        if(randSpec>=time): # wav is too short, use the whole wav.
            spec_mag = mag_T
        else:
            randStart = np.random.randint(0, time-randSpec)
            spec_mag = mag_T[:, randStart:randStart+randSpec]

        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)

    else:
        # preprocessing, subtract mean, divided by time-wise var
        mu = np.mean(spec_mag, 0, keepdims=True)
        std = np.std(spec_mag, 0, keepdims=True)
        spec_mag = (spec_mag - mu) / (std + 1e-5)

    return spec_mag


class AudioProducer(object):

    def __init__(self, data_json, batch_size, sample_rate=16000,
                 min_duration=600, max_duration=2500):
        """
        Args:
            data_json : json format file with speech data.
                        'path', the path of the wave file.
                        'spkid', the speaker's identity in int.
            batch_size : Size of the batches for training.
            sample_rate : Rate to resample audio prior to feature computation.
            min_duration : Minimum length of audio sample in milliseconds.
            max_duration : Maximum length of audio sample in milliseconds.
        """
        self.batch_size = batch_size
        self.sample_rate = sample_rate
        self.min_duration = min_duration
        self.max_duration = max_duration
        with open(data_json, 'r') as fid:
            self.data = json.load(fid)


    def queue_featurize(self, consumer, producer, sample_rate):
        while True:
            try:
                batch = consumer.get(block=True, timeout=5)
            except mp.queues.Empty as e:
                print("queue_featurize finished or error happened.")
                return
            labels = []
            inputs = []
            rand_duration = np.random.randint(self.min_duration, self.max_duration)
            minSpec = rand_duration
            utterances = []
            for b in batch:
                labels.append(int(b['spkid']))
                utterance_spec = load_data(b['path'], 
                                            sr=sample_rate,
                                            rand_duration=rand_duration,
                                            is_training=True)
                if(utterance_spec is None): # Error loading some audio file.
                    break
                if(utterance_spec.shape[1]<minSpec): # align to mini audio file.
                    minSpec = utterance_spec.shape[1]
                utterances.append(utterance_spec)

            if(len(utterances)!=len(batch)): # Error loading some audio file.
                continue
            else:
                for utterance_spec in utterances:
                    inputs.append(np.expand_dims(utterance_spec[:,:minSpec], -1))

            producer.put((np.array(inputs), labels))


    def iterator(self, max_size=512, num_workers=10):
        random.shuffle(self.data)
        batches = [self.data[i:i+self.batch_size]
                   for i in range(0, len(self.data) - self.batch_size + 1, self.batch_size)]

        consumer = mp.Queue()
        producer = mp.Queue(max_size)
        for b in batches:
            consumer.put(b)

        procs = [mp.Process(target=self.queue_featurize,
                            args=(consumer, producer,
                                  self.sample_rate))
                 for _ in range(num_workers)]
        for p in procs:
            p.start()

        for _ in batches:
            yield producer.get()


