# AdaptiveSampling-DRL

## Overview
This repo describes the official software package developed for and used to create the free and public adaptive sampling rate controller. It is a a runtime implementation of sampling rate control using deep reinforcement learning (DRL). By using essential morphological details contained in the heartbeat waveform, the DRL agent can control the sampling rate and effectively reduce energy consumption at runtime.

## Initial Setup
## Reading the MIT-BIH data
#### Since the data got from [MIT-BIH](https://physionet.org/content/mitdb/1.0.0/) is not segmented and normalized, the [data_prep_MIT-BIH_beat.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/data_prep_MIT-BIH_beat.py) should be run.
## Running Scripts

## Motivational Example
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/dana_MIT_constant.py). Requirements_2 is needed to run the script without errors.
<img src="./Figures/mot_jbhi.jpg" width="600">

## Model Architecture
#### The model architecture is implemented in [models.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/models.py). Requirements_2 is needed to run the script without errors.
<img src="./Figures/jbhi_arch.png" width="600">


## DRL Training
#### Before training the DRL Agent, please fill in the necessary fields for the configuration file [DQL.yaml](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/DRL/configs/DQL.yaml)
#### To train the DRL agent, run [DQL_trainer.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/DRL/src/DQL_trainer.py) which will take the configuration settings mentioned previously.

## DRL Inference
#### The same configuration file is used for evaluating the DRL agent. The evaluation results referenced in the paper can be obtained by running the [experiment.py](https://github.com/Berken-demirel/AdaptiveSampling-DRL/blob/main/DRL/src/experiment.py) file.
