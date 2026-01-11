# AdaptiveSampling-DRL

## Acknowledgements and References

This codebase was developed as part of the Reinforcement Learning module coursework project.  
The implementation is inspired by and partially based on the methodology and open-source implementation described in:

> Demirel, B. U., Chen, L., & Al Faruque, M. A. (2023).  
> *Data-driven Energy-efficient Adaptive Sampling Using Deep Reinforcement Learning*.  
> ACM Transactions on Computing for Healthcare, 4(3).

The original paper proposes a deep reinforcement learning framework for adaptive ECG sampling to reduce energy consumption while maintaining arrhythmia classification performance.  
This repository is an educational re-implementation and extension of those ideas for coursework purposes and is **not an official reproduction** of the original authors’ code.

## Overview
This repo describes the official software package developed for and used to create the free and public adaptive sampling rate controller. It is a a runtime implementation of sampling rate control using deep reinforcement learning (DRL). By using essential morphological details contained in the heartbeat waveform, the DRL agent can control the sampling rate and effectively reduce energy consumption at runtime.

## Initial Setup
## Reading the MIT-BIH data
#### Since the data got from [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) is not segmented and normalized, the [data_prep_MIT-BIH_beat.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/data_prep_MIT-BIH_beat.py) should be run.
## Running Scripts

## Motivational Example
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/dana_MIT_constant.py). Requirements_2 is needed to run the script without errors.


## Model Architecture
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/models.py). Requirements_2 is needed to run the script without errors.



## DRL Training
#### Before training the DRL Agent, please fill in the necessary fields for the configuration file [DQL.yaml](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/DRL/configs/DQL.yaml)
#### To train the DRL agent, run [DQL_trainer.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/DRL/src/DQL_trainer.py) which will take the configuration settings mentioned previously.

## DRL Inference
#### The same configuration file is used for evaluating the DRL agent. The evaluation results referenced in the paper can be obtained by running the [experiment.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/DRL/src/experiment.py) file.
