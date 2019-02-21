# End-to-end Automatic Speech Recognition Systems - PyTorch Implementation

This is an open source project (formerly named **Listen, Attend and Spell - PyTorch Implementation**) for end-to-end ASR implemented with Pytorch, the well known deep learning toolkit.

The end-to-end ASR was based on Listen, Attend and Spell<sup>[1](#Reference)</sup>. Multiple techniques proposed recently were also implemented, serving as additional plug-ins for better performance. For the list of techniques implemented, please refer to the [highlights](#Highlights), [configuration](config/) and [references](#Reference).

Feel free to use/modify them, any bug report or improvement suggestion will be appreciated. If you have any questions, please contact r07922013[AT]ntu.edu.tw. If you find this project helpful for your research, please do consider to cite [my paper](#Citation), thanks!

## Highlights

![](log/demo.png)

- Acoustic feature extraction
    - Purepython extraction using librosa as the backend
    - One-click execution scripts (currently supporting TIMT & LibriSpeech)
    - Phoneme/character/subword<sup>[2](#Reference)</sup>/word embedding for text encoding

- End-to-end ASR 
    - Seq2seq ASR with different types of encoder/attention<sup>[3](#Reference)</sup>
    - CTC-based ASR<sup>[4](#Reference)</sup>, which can also be hybrid<sup>[5](#Reference)</sup> with the former
    - *yaml*-styled model construction and hyper parameters setting
    -  Training process visualization with [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard), including attention alignment

- Speech recognition (decoding)
    - Beam search decoding
    - RNN language model training and joint decoding for ASR<sup>[6](#Reference)</sup>
    - Joint CTC-attention based decoding<sup>[6](#Reference)</sup>


## Requirements

- Python 3
- Computing power (high-end GPU) and memory space (both RAM/GPU's RAM) is **extremely important** if you'd like to train your own model.
- Required packages and their use are listed [here](used_package.txt).

## Instructions


***Before you start, make sure all the [packages required](used_package.txt) were installed correctly***

### Step 0. Preprocessing - Acoustic Feature Extraction & Text Encoding

Preprocessing scripts are placed under [`data/`](data/), you may execute them directly. The extracted data, which is ready for training, will be stored in the same place by default. For example,
```
cd data/
python3 preprocess_libri.py --data_path <path to LibriSpeech on your computer> 
```

The parameters available for these scripts are as follow,

| Options        | Description                                                               |
|-------------------|---------------------------------------------------------------------------|
| data_path         | Path to the raw dataset (can be obtained by download & unzip)                 |
| feature_type      | Which type of acoustic feature to be extracted, fbank or mfcc                                                             |
| feature_dim       | Feature dimension, usually depends on the feature type (e.g. 13 for mfcc) |
| apply_delta       | Append delta of the acoustic feature to itself                            |
| apply_delta_delta | Append delta of delta                                                     |
| apply_cmvn        | Normalize acoustic feature                                                |
| output_path       | Output path for extracted data (by default, it's data/)                   |
| target            | Text encoding target, one of phoneme/char/subword/word                    |
| n_tokens          | Vocabulary size, only applies for subword/word    

You may check the parameter type and default value by using the option ```--help``` for each script.

### Step 1. Configuring - Model Design & Hyperparameter Setup

All the parameters related to training/decoding will be stored in a yaml file. Hyperparameter tuning and massive experiment and can be managed easily this way. See [documentation and examples](config/) for the exact format. **Note that the example configs provided were not fine-tuned**, you may want to write your own config for best performance.

### Step 2. Training - End-to-end ASR (or RNN-LM) Learning

Once the config file is ready, run the following command to train end-to-end ASR (or language model)
```
python3 main.py --config <path of config file> 
```
All settings will be parsed from the config file automatically to start training, the log file can be accessed through TensorBoard. ***Please notice that the error rate reported on the TensorBoard is biased (see issue #10), you should run the testing phase in order to get the true performance of model***. For example, train an ASR on LibriSpeech and watch the log with
```
python3 main.py --config config/libri_example.yaml
# open TensorBoard to see log
tensorboard --logdir log/
# Train an external language model
python3 main.py --config config/libri_example_rnnlm.yaml --rnnlm
```
There are also some options,  which do not influence the performance (except `seed`), are available in this phase including the followings

| Options | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| config  | Path of config file                                                                           |
| seed    | Random seed, **note this is an option that affects the result**                                         |
| name    | Experiments for logging and saving model, by default it's <name of config file>_<random seed> |
| logdir  | Path to store training logs (log files for tensorboard), default `log/`                                                   |
| ckpdir  | Path to store results, default `result/<name>`                                                |
| njobs   | Number of workers for Pytorch DataLoader                                                      |
| cpu     | CPU-only mode, not recommended                                                                |
| no-msg  | Hide all message from stdout                                                                  |
| rnnlm   | Switch to rnnlm training mode                                                               |
| test    | Switch to decoding mode (do not use during training phase) 


### Step 3. Testing - Speech Recognition & Performance Evaluation

Once a model was trained, run the following command to test it
```
python3 main.py --config <path of config file> --test
```
Recognition result will be stored at `result/<name>/` as a txt file with auto-naming according to the decoding parameters specified in config file. The result file may be evaluated with `eval.py`. For example, test the ASR trained on LibriSpeech and check performance with
```
python3 main.py --config config/libri_example.yaml --test
# Check WER/CER
python3 eval.py --file result/libri_example_sd0/decode_*.txt
```
**Notice that the meaning of options for `main.py` in this phase will change**

| Options | Description                                                                                   |
|---------|-----------------------------------------------------------------------------------------------|
| test    | *Must be enabled*|
| config  | Path of config file                                                                           |
| name    |*Must be identical to the models training phase*  |
| ckpdir  | *Must be identical to the models training phase*                                               |
| njobs   | Number of threads used for decoding, very important in terms of efficiency. Large value equals fast decoding yet RAM/GPU RAM expensive.    |
| cpu     | CPU-only mode, not recommended                                                     |
| no-msg  | Hide all message from stdout                                                              |

Rest of the options are *ineffective*  in the testing phase.

## ToDo

- Plot attention map during testing
- Pure CTC training 
- Resume model training

## Acknowledgements 
- Parts of the implementation refer to [ESPnet](https://github.com/espnet/espnet), a great end-to-end speech processing toolkit by Watanabe *et al*.
- Special thanks to [William Chan](http://williamchan.ca/), the first author of LAS, for answering my questions during implementation.
- Thanks [xiaoming](https://github.com/lezasantaizi), [Odie Ko](https://github.com/odie2630463) and [b-etienne](https://github.com/b-etienne) for identifying several issues in our implementation. 

## Reference

1. [Listen, Attend and Spell](https://arxiv.org/abs/1508.01211v2), W Chan *et al.*
2. [Neural Machine Translation of Rare Words with Subword Units](http://www.aclweb.org/anthology/P16-1162), R Sennrich *et al.*
3. [Attention-Based Models for Speech Recognition](https://arxiv.org/abs/1506.07503), J Chorowski *et al*.
4. [Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks](https://www.cs.toronto.edu/~graves/icml_2006.pdf), A Graves *et al*.
5. [Joint CTC-Attention based End-to-End Speech Recognition using Multi-task Learning](https://arxiv.org/abs/1609.06773), S Kim *et al.* 
6.  [Advances in Joint CTC-Attention based End-to-End Speech Recognition with a Deep CNN Encoder and RNN-LM](https://arxiv.org/abs/1706.02737), T Hori *et al.* 

## Citation

```
@inproceedings{liu2019adversarial,
  title={Adversarial Training of End-to-end Speech Recognition Using a Criticizing Language Model},
  author={Liu, Alexander and Lee, Hung-yi and Lee, Lin-shan},
  booktitle={Acoustics, Speech and Signal Processing (ICASSP)},
  year={2019},
  organization={IEEE}
}
```
