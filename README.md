# AmbientAttack: Black-Box Attacks on Automatic Speech Recognition via Acoustic Injection

This paper has been submitted to ICASSP 2027 for peer reviewing. <br>

Our demo website [[demo]](https://anonymous.4open.science/r/ambientattack-469F/index.html)

# How to Use
```
conda create -n ambientattack python=3.9.21
conda activate ambientattack
pip install -r requirements.txt
```

## Download Wav2Vec2.0
- Please clone this website https://github.com/facebookresearch/fairseq/tree/main/examples/wav2vec
- Download the checkpoint Wav2Vec 2.0 Base
- dict.ltr.txt file: https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt

## Download NISQA
- Please clone this website https://github.com/gabrielmittag/NISQA
- Use the checkpoint nisqa_mos_only.tar

## Dataset
- Download ESC-50 dataset: https://github.com/karolpiczak/ESC-50
- Resample the environment sound to 16kHz

We also organize the checkpoints and ESC-50 dataset with 16kHz: [[Materials]](https://drive.google.com/drive/folders/1GCCkmeuq3TzzW92_n43s9gezy7dei6gj?usp=sharing)

# Attack -- Example for Wav2Vec
Run the first step Ambient Sound Selection
```
python ambientattack_first_step_wav2vec.py \
  --speech ./data_test/speech/01_sample/01_sample.wav \
  --environment-pool ./dataset/ESC-50-master/audio_16khz \
  --wav2vec-model ../wav2vec/checkpoint/wav2vec_small_960h.pt \
  --wav2vec-dictionary ../wav2vec/checkpoint/dict.ltr.txt
```

Run the second step Ambient Sound Perceptual Optimization (You can directly use the test sample in the folder "data_test" without running first step)
```
python ambientattack_second_step_wav2vec.py \
  --speech ./data_test/speech/01_sample/01_sample.wav \
  --environment ./data_test/environment/01_sample/2-59565-A-46.wav \
  --reference-file ./data_test/speech/01_sample/01_sample.txt \
  --wav2vec-model ../wav2vec/checkpoint/wav2vec_small_960h.pt \
  --wav2vec-dictionary ../wav2vec/checkpoint/dict.ltr.txt \
  --nisqa-root ../NISQA \
  --nisqa-model ../NISQA/weights/nisqa_mos_only.tar
```
