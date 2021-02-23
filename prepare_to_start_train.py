import argparse
import os
from utils.utils import print_one_line, print_all_properties
from utils.__init__ import setup_environment
from configs.config import Config
from datasets.ResArch_dataset import ResArch_Dataset
setup_environment()

# these lines are for reading flags
parser = argparse.ArgumentParser(prog="Speaker Recognition training")
DEFAULT_YAML = './config.yml'
parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
parser.add_argument("--model", type=str, default='ResArch', help='Which model Do you want to train on?(SpeakerNet/ResArch/SincNet)')
args = parser.parse_args()


'''
dataset_root: str,
                my_audios_folder : str,
                my_noises_folder : str,
                valid_split: float,
                shuffle_seed : int,
                sample_rate: int,
                sacle: float,
                batch_size : int,
                epochs : int,
                stage : str,
                shuffle: bool = False

'''


def train_ResArch(config):
  print(os.path.abspath("./"))
  my_ds_object = ResArch_Dataset(config.learning_config.dataset_root,
                                 config.learning_config.my_audios_folder,
                                 config.learning_config.my_noises_folder,
                                 config.learning_config.valid_split,
                                 config.learning_config.shuffle_seed,
                                 config.speech_config['sample_rate'],
                                 config.speech_config['scale'],
                                 config.learning_config.running_config.batch_size,
                                 config.learning_config.running_config.num_epochs,
                                 config.learning_config.train_dataset_config.stage,
                                 config.learning_config.train_dataset_config.shuffle,)

  # print_all_properties(my_ds_object)                            
  my_ds_object.prepare_tf_dataset_obj()
  for i in my_ds_object.get_train_ds().take(3):
    print(i)
# read Config file.
if args.model == "ResArch":
    config = Config(args.config)
    print("You selected to use ResArch")
    train_ResArch(config)

elif args.model == "SincNet":
    print("you selected SincNet. NOT IMPLEMENTED YET")
elif args.model == "SpeakerNet":
    print("you selected SpeakerNet. NOT IMPLEMENTED YET")





