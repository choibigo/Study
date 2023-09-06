import argparse
import os
import json
import shutil

import torch

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='Siamese Network')

# data params
data_arg = parser.add_argument_group('Data Params')
data_arg.add_argument('--valid_trials', type=int, default=320,
                      help='# of validation 1-shot trials')
data_arg.add_argument('--test_trials', type=int, default=400,
                      help='# of test 1-shot trials')
data_arg.add_argument('--way', type=int, default=20,
                      help='Ways in the 1-shot trials')
data_arg.add_argument('--num_train', type=int, default=90000,
                      help='# of images in train dataset')
data_arg.add_argument('--batch_size', type=int, default=128,
                      help='# of images in each batch of data')
data_arg.add_argument('--num_workers', type=int, default=4,
                      help='# of subprocesses to use for data loading')
data_arg.add_argument('--pin_memory', type=str2bool, default=False,
                      help='Whether to save the pin memory')
data_arg.add_argument('--shuffle', type=str2bool, default=True,
                      help='Whether to shuffle the dataset between epochs')
data_arg.add_argument('--augment', type=str2bool, default=False,
                      help='Whether to use data augmentation for train data')

# training params
train_arg = parser.add_argument_group('Training Params')
train_arg.add_argument('--epochs', type=int, default=200,
                       help='# of epochs to train for')
train_arg.add_argument('--init_momentum', type=float, default=0.5,
                       help='Initial layer-wise momentum value')
train_arg.add_argument('--lr', type=float, default=3e-4,
                       help='learning rate')
train_arg.add_argument('--train_patience', type=int, default=20,
                       help='Number of epochs to wait before stopping train')
train_arg.add_argument('--optimizer', type=str, default="Adam",
                       help='Select optimizer "Adam" or "SGD"')

# other params
misc_arg = parser.add_argument_group('Misc.')
misc_arg.add_argument('--flush', type=str2bool, default=False,
                      help='Whether to delete ckpt + log files for model no.')
misc_arg.add_argument('--num_model', type=str, default="1",
                      help='Model number used for unique checkpointing')
misc_arg.add_argument('--use_gpu', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--best', type=str2bool, default=True,
                      help='Load best model or most recent for testing')
misc_arg.add_argument('--seed', type=int, default=1,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--data_dir', type=str, default='./data/processed/',
                      help='Directory in which data is stored')
misc_arg.add_argument('--logs_dir', type=str, default='./result/',
                      help='Directory in which logs wil be stored')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')


def get_config():
    config, _ = parser.parse_known_args()

    assert config.num_workers > 0, f"number of worker must be >= 1, you are {config.num_workers}"
    assert int(config.num_model) > 0, f"number of model must be >= 1, you are {config.num_model}"

    if config.use_gpu and torch.cuda.is_available():
        print(f"[*] use GPU ", end="")
        device_count = torch.cuda.device_count()
        for id in range(device_count):
            if id == device_count - 1:
                print(torch.cuda.get_device_name(id))
            else:
                print(f'{torch.cuda.get_device_name(id)}', end="")

        torch.cuda.manual_seed(config.seed)
        config.num_workers: 1
        config.pin_memory: True

    if config.resume:
        config.best = False

    config.logs_dir = os.path.join(config.logs_dir, config.num_model)

    return config


def save_config(config):
    param_path = os.path.join(config.logs_dir, 'params.json')

    if not os.path.isfile(param_path):
        print(f"Save params in {param_path}")

        all_params = config.__dict__
        with open(param_path, 'w') as fp:
            json.dump(all_params, fp, indent=4, sort_keys=True)
    else:
        print(f"[!] Config file already exist.")
        raise ValueError


def load_config(config):
    param_path = os.path.join(config.logs_dir, 'params.json')
    params = json.load(open(param_path))

    config.__dict__.update(params)

    config.resume = True

    return config


def prepare_dirs(config):
    path = config.logs_dir
    if not os.path.exists(path):
        os.makedirs(os.path.join(path, 'logs'))
        os.makedirs(os.path.join(path, 'models'))
    if config.flush:
        shutil.rmtree(path)
        os.makedirs(os.path.join(path, 'logs'))
        os.makedirs(os.path.join(path, 'models'))


if __name__ == '__main__':
    get_config()
