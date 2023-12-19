## Used dataset
In this thesis, the fastMRI dataset is used.

Important stuff we need to comply with:

> a. In any published abstract, I will cite “NYU fastMRI” as the source of the data in the abstract.

> b. In any published manuscripts using data from NYU fastMRI, I will reference the following paper: https://pubs.rsna.org/doi/10.1148/ryai.2020190007

> c. I will include language similar to the following in the methods section of my manuscripts in order to accurately acknowledge data source: "Data used in the preparation of this article were obtained from the NYU fastMRI Initiative database (fastmri.med.nyu.edu).[citation of Knoll et al] As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at: fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images".


## Make sure you use a GPU when running a process
The micgpu command only ensures that if you use GPU, this is done on the desired GPU. Luckily, when using tensorflow, models will transparently run on a single GPU with no code changes required IF the tensorflow's required CUDA and cuDNN versions are installed and added to the PATH and LD_LIBRARY_PATH variables. If this is not the case, CUDA will throw an error and start using a CPU. The following set-up is known to work for tensorflow 2.2.0:

> 1. echo $PATH: 
>
> /opt/ANTs/bin:/usr/local/fsl/bin:/usr/local/cuda-10.1:/SOFTWARE/scripts:/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/conda/envs/DL_MRI_reconstruction/bin:/opt/anaconda/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin

> 2. echo $LD_LIBRARY_PATH: 
>
> /usr/local/cuda-10.1/targets/x86_64-linux/lib

Changing these variables (making additions to them) can be done by running e.g. the following commands in the command shell:

export PATH=/opt/ANTs/bin:/usr/local/fsl/bin:/usr/local/cuda-10.1:/SOFTWARE/scripts:/usr/local/micapollo01/MIC/DATA/STUDENTS/mvhave7/conda/envs/DL_MRI_reconstruction/bin:/opt/anaconda/anaconda3/condabin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/snap/bin

export LD_LIBRARY_PATH=/usr/local/cuda-10.1/targets/x86_64-linux/lib
