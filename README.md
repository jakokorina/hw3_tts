# TTS project

## Installation guide

1. Clone this repository

```shell
git clone https://github.com/jakokorina/hw3_tts.git
cd hw3_tts 
```

2. Install requirements

```shell
pip install -r ./requirements.txt
```

3. Download waveglow. Here's example on python but you can use
   anything you want

```python
import gdown

url = "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j"
output = "mel.tar.gz"
gdown.download(url, output, quiet=False)
```

Then place it in right place

```shell
mkdir hw_tts/waveglow/pretrained_model/
mv waveglow_256channels_ljs_v2.pt hw_tts/waveglow/pretrained_model/waveglow_256channels.pt
```

4. Download my model checkpoint

```python
import gdown

url = "https://drive.google.com/u/0/uc?id=1zdxuCP1-szvx5TkKmSsAgAAQYE6VjuPx"
output = "model.pth"
gdown.download(url, output, quiet=False)
```

5. If you want to train then download additional files:

- Alignments:

```shell
wget https://github.com/xcmyz/FastSpeech/raw/master/alignments.zip
unzip alignments.zip
```

- MELS

```python
import gdown

url = "https://drive.google.com/u/0/uc?id=1cJKJTmYd905a-9GFoo5gKjzhKjUVj83j"
output = "mel.tar.gz"
gdown.download(url, output, quiet=False)
```

```shell
tar -xvf mel.tar.gz
```

- LJSpeech:

```python
import gdown

url = "https://drive.google.com/u/0/uc?id=1-EdH0t0loc6vPiuVtXdhsDtzygWNSNZx"
output = "train.txt"
gdown.download(url, output, quiet=False)
```

```shell
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2 -o /dev/null
mkdir data
tar -xvf LJSpeech-1.1.tar.bz2
mv LJSpeech-1.1 data/LJSpeech-1.1
mv train.txt data/
```

- Pitch and energy files preprocessed by me:

```python
import gdown

url = "https://drive.google.com/uc?id=1-3aBBRfI1e9s4mzZLNbMyV-qiJ2T32IA"
output = "pitch.tar.gz"
gdown.download(url, output, quiet=False)

url = "https://drive.google.com/uc?id=11ZURBIrG6JDJpC_SMSWpaQ4GXaDG1I2d"
output = "energy.tar.gz"
gdown.download(url, output, quiet=False)
```

```shell
tar -xvf pitch.tar.gz
tar -xvf energy.tar.gz
```

## What I did

I took the code from our [seminar](https://github.com/markovka17/dla/blob/2022/week07/FastSpeech_sem.ipynb), 
split it into files using asr_project_template. Then I replaced PreLayerNorm with Post LN, log(duration)
instead of duration, and trained it. After that I computed pitch and energy embeddings, added them to training 
pipeline and trained FastSpeech2. All modifications are available in commit history of this repo.
All training logs can be found [here](https://wandb.ai/jakokorina/tts_project/overview?workspace=user-jakokorina).
Note that WandB loggs audio with noise, in real audio there's no noise.
WandB had some problems, that's why not all charts contain all data, but it can be found in the logs.

The hardest thing was to reformat code and understand mechanism of energy and pitch
predictions. 

### Pitch and energy

At first I computed pitch and energy during in `get_data_to_buffer` function like this: 

```
wav_files = sorted([os.path.join("data/LJSpeech-1.1/wavs/", f) for f in os.listdir("data/LJSpeech-1.1/wavs/") \
                        if os.path.isfile(os.path.join("data/LJSpeech-1.1/wavs/", f))])


wav_path = wav_files[i]
wav, _ = librosa.load(wav_path)
pitch, t = pw.dio(
    wav.astype(np.float64),
    hparams_audio.sampling_rate,
    frame_period=hparams_audio.hop_length / hparams_audio.sampling_rate * 1000,
)
pitch = pw.stonemask(wav.astype(np.float64), pitch, t, 22050)
pitch = torch.tensor(pitch[: sum(duration)])

energy = torch.stft(torch.tensor(wav),
                    n_fft=1024,
                    hop_length=256,
                    win_length=1024
                    ).transpose(0, 1)
energy = torch.linalg.norm(torch.sqrt(energy[:, :, 0] ** 2 + energy[:, :, 1] ** 2), dim=-1)
```

But then I realized that this computation takes a lot of time, so I decided to
save `energy` and `pitch` from above into files using `torch.save` and then load 
them from files. Loading pipeline can be found in my repo. 

### FastSpeech2 w/o energy and pitch

I don't have resulting files for this model because I thought that WandB audio logging
is correct. But I have all checkpoints and I can give them on request. 

## Results

They are located in the `results` folder. I used the following sentences:

```
"A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest",
"Massachusetts Institute of Technology may be best known for its math, science and engineering education",
"Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space"
```

I attached TTS versions of this sentences in different modes:
- Default mode (_default)
- +- 20% of speed(s), pitch(p) and energy(e)
- +- 20% of speed, pitch and energy all together(_all)

The  number in the name of .wav corresponds to index of sentence in the list above.

## Reproducing

- Training
```shell
python3 train.py -c hw_tts/config.json
```
- Testing
```shell
python3 test.py -c hw_tts/config.json -r model.pth
```

Note that if you want to change the frases you can pass a list of  sentences into the
`synthesis.utils.get_data()` on line 41 of `test.py`.
