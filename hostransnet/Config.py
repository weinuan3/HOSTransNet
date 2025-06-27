"""
Our experimental codes are based on
https://github.com/McGregorWwww/UCTransNet
We thankfully acknowledge the contributions of the authors
"""

import os
import torch
import time
import ml_collections

## PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"        # change this as needed
use_cuda = torch.cuda.is_available()
seed = 888
os.environ['PYTHONHASHSEED'] = str(seed)
n_filts = 32            # change this to train larger ACC-UNet model
cosineLR = True         # whether use cosineLR or not
n_channels = 3
n_labels = 1
epochs = 2000
img_size = 224
print_frequency = 1
save_frequency = 100
vis_frequency = 100
early_stopping_patience = 50


task_name = 'GlaS'
# task_name = 'DSB2018'
# task_name = 'isbi2014'

learning_rate = 1e-5
batch_size = 4

# model_name = 'UKAN'
# model_name = 'UNet'
# model_name = 'UnetPlusPlus'
# model_name = 'SwinUnet'
# model_name = 'MultiResUnet'
# model_name = 'UCTransNet'
# model_name = 'ACC_UNet'
# model_name = 'AttentionUNet'
# model_name = 'DANet'
model_name = 'HOSTransNet'
# model_name = 'Segformer'
# model_name = 'RollingUnet'
# model_name = 'cmunext'

train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'
session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          = task_name +'/'+ model_name +'/' + session_name + '/'
model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'


def get_Mynet_config():
    config = ml_collections.ConfigDict()
    config.transformer = ml_collections.ConfigDict()
    config.KV_size = 960  # KV_size = Q1 + Q2 + Q3 + Q4
    config.transformer.num_heads = 1
    config.transformer.num_layers = 4
    config.expand_ratio = 4
    config.patch_sizes = [16, 8, 4, 2]
    config.base_channel = 64  # base channel of U-Net
    config.n_classes = 1

    # ********** unused **********
    config.transformer.embeddings_dropout_rate = 0.1
    config.transformer.attention_dropout_rate = 0.1
    config.transformer.dropout_rate = 0
    return config

# used in testing phase, copy the session name in training phase
test_session = "Test_session_05.04_00h57"