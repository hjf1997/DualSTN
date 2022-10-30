# Decoupling long- and short-term patterns in spatiotemporal inference


## Requirements
Anaconda is highly recommended, and the code is tested under:
* Python = 3.6
* Numpy = 1.19 
* Pytorch = 1.7
* Pandas = 1.1
* tqdm = 4.49

## Datasets
The datasets adopted in our work are accessible at:
* METR-LA: [https://github.com/liyaguang/DCRNN]().
* PeMS-Bay: [https://github.com/liyaguang/DCRNN]().
* NREL: [https://www.nrel.gov/grid/solar-power-data.html]().
* Beijing-Air: [https://www.biendata.xyz/competition/kdd_2018/data](). (Register is required)

## Run the program
### Arguments
Key arguments are listed below:

setting | default | values | help
:--:|:--:|:--:|:--:
--name|DualSTN|DualSTN| name of the experiment. It partially decides checkpoint file name.
--gpu_ids|-1| int | index of GPU. Multiple GPUs are not supported.
--file_time|N.A.|str| time when the model was trained. specify during evaluation. 
--dataset_mode|metrla|metrla, nrel, bjair, pemsbay| choose what dataset is used. 
--epoch | N.A. | int| which epoch to load during evaluation. 

### Model Training and Evaluation
Here are the command for training the model on METR-LA. The framework will automatically evaluate the best checkpoint on the validation/testing set and save all related results to a .pkl file.
```
python train.py --name DualSTN --dataset_mode metrla --gpu_ids 0
```
Alternatively you can use [train.sh](./train.sh) to train the model with different initialization seeds.
```
train.sh DualSTN metrla 0 2030
```
Assume the timestamp of running the code is 20210725T160246, the framework will store configurations and  checkpoints at
```
DualSTN/
│   
└───checkpoints/
   │
   └───metrla/
       │   
       └──DualSTN_20210725T160246 
```
For other functions like continue training, please refer the code at (base_options.py)

### Pretrained checkpoints
We provide the pretrained checkpoints for METR-LA, PeMS-Bay, NREL, and Beijing-Air. The checkpoints are stored at [checkpoints](https://drive.google.com/drive/folders/1TjaqX_2xi1hVkFwWhah9NIIUGLmpciZf?usp=sharing). The checkpoints are named as `DualSTN_{dataset}_{timestamp}`. For example, the checkpoint for METR-LA is named as `DualSTN_metrla_20210725T160246`. The checkpoints are trained with the same hyperparameters as in the paper.
## Acknowledge
This framework is largely based on [CycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).
Some parts of the code are adopted from [IGNNK](https://github.com/Kaimaoge/IGNNK).
Thanks for their great works! :)