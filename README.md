## Used dataset
In this thesis, the fastMRI dataset is used.

Important stuff we need to comply with:

> a. In any published abstract, I will cite “NYU fastMRI” as the source of the data in the abstract.

> b. In any published manuscripts using data from NYU fastMRI, I will reference the following paper: https://pubs.rsna.org/doi/10.1148/ryai.2020190007

> c. I will include language similar to the following in the methods section of my manuscripts in order to accurately acknowledge data source: "Data used in the preparation of this article were obtained from the NYU fastMRI Initiative database (fastmri.med.nyu.edu).[citation of Knoll et al] As such, NYU fastMRI investigators provided data but did not participate in analysis or writing of this report. A listing of NYU fastMRI investigators, subject to updates, can be found at: fastmri.med.nyu.edu. The primary goal of fastMRI is to test whether machine learning can aid in the reconstruction of medical images".



## Model versions in this repository
Currently, there are two code versions of GrappaNet:

> 1. model_v1: This version contains the original (personalized) GrappaNet code which loads all training data in RAM memory when training, which causes RAM issues, even if a smaller dataset is used (e.g. training is possible with 25 files, but only 1 training epoch is possible before the program is out of memory). A first step towards optimal RAM usage is splitting the preprocessing step from the training step in the GrappaNet pipeline, which is done here. Still, RAM issues persist.

> 2. model_v2: This version contains GrappaNet code which counters the RAM issues from model_v1 by loading the training data in batches while training. The preprocessing step is also modified accordingly. This circumvents the need to load all training data in RAM memory when training.

