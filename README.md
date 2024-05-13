# Master's Thesis - Mathijs Vanhaverbeke
Code base of my master's thesis (2023-2024), submitted for the degree of M.Sc. in Biomedical Engineering, option Biomedical Data Analytics
Title: Influence of hybrid modelling on deep learning-based MRI reconstruction performance


## Used dataset
In this thesis, the fastMRI dataset is used. The used, preprocessed version of the fastMRI data can be found on the location: /usr/local/micapollo01/MIC/DATA/SHARED/NYU_FastMRI/Preprocessed/


## Contributions
During the past year, multiple models and methods were explored. Ultimately, the code in this repository is related to four things:
- fastMRI data exploration
- Evaluation metric implementations
- Validation of DeepMRIRec
- Hybrid Neural Network comparisons


## Repository structure
Related to the above mentioned explored topics, this repository contains multiple subfolders:
- BaselineUNet
- CS
- CSUNet
- DeepMRIRec
- fastMRI
- Grappa
- GrappaUNet
- ModelWeightsComparison
- Overfitexperiments
- Sense
- SenseUNet
- StatisticalTests
- ZeroFilled
- ZeroFilledNoACS
- ZeroFilledNoACSCS


The folders BaselineUNet, CS, CSUNet, Grappa, GrappaUNet, Sense, SenseUNet, ZeroFilled, ZeroFilledNoACS, and ZeroFilledNoACSCS contain all the code necessary for the HNN comparisons made in the thesis manuscript. Those folders related to a deep learning model also contain a Checkpoint folder containing the trained model used to generate the presented results. The BaselineUNet even contains multiple saved checkpoints, related to different training scenarios. For example, there is a BaselineUNet version trained only on R=4, on both R=4+8, on both R=4+8 with L2 regularization, and there is also a BaselineUNet version trained on SENSE masks instead of GRAPPA masks. Additionally, there is code to let BaselineUNet make predictions on R=3 data. Similarly, also GrappaUNet has two checkpoints: one for a regularized training and one for an unregularized training. Other models have one checkpoint, resulting from a regularized training. Importantly, each of these aformentioned main folders contains a .txt file. In this file, more info can be found regarding which conda environment to use and regarding which source material was used to write the code. Folders containing the code of Hybrid Neural Networks also contain the code that was used to preprocess the fastMRI data, saving time during training.


The folders fastMRI, ModelWeightsComparison, Overfitexperiments, and StatisticalTests are all related to data and results analyses. These mostly contain notebooks, with the content speaking for itself.


Lastly, the folder DeepMRIRec contains the used code for the training of DeepMRIRec. This folder contains the code for multiple variations of its architecture, and multiple variations of the used dataloader. When wanting to run the model on micsd01, the small GPU version of the model needs to be used, as the original version is quite heavy.


## Training the HNNs
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run 'micgpu 6 python UNet.py --mode train --challenge multicoil --mask_type equispaced --center_fractions 0.08 0.04 --accelerations 4 8'


## HNN model inference
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run 'micgpu 6 python UNet.py --mode test --challenge multicoil --mask_type equispaced --resume_from_checkpoint checkpoint-path'
- Change checkpoint-path to the correct path


## HNN inference evaluations
This is done through the commandline, for example:
- Go to the folder of the desired HNN
- Activate the correct conda env, listed in the .txt file
- Run 'micgpu 6 python evaluate_with_vgg_and_mask.py --target-path path-1 --predictions-path path-2 --challenge multicoil --acceleration 4'
- Change path-1 and path-2 to the correct paths


## Other examples
Other examples of how to use the commandline to run the code can often be found on the repositories linked in the .txt files when necessary. Alternatively, the required input arguments to run a python file, and what they do, can also be deduced by looking at the code.


## Training DeepMRIRec
The files pertaining to the validation of DeepMRIRec can be found in the DeepMRIRec folder. Make sure you use a GPU when running a DeepMRIRec training process. The micgpu command only ensures that if you use GPU, this is done on the desired GPU number. Luckily, when using tensorflow, models will transparently run on a single GPU with no code changes required if the tensorflow's required CUDA and cuDNN versions are installed and added to the PATH and LD_LIBRARY_PATH variables. If this is not the case, CUDA will throw an error and start using a CPU. The following set-up is known to work for tensorflow 2.2.0, used by the DL_MRI_reconstruction conda environment compatible with the code of DeepMRIRec:
- In your terminal run 'source cuda_settings.sh'


Note that this is not necessary for the HNNs' code, as the conda environments automatically handle GPU usage there.

