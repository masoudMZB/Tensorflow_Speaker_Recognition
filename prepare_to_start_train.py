import argparse
import os
from utils.utils import print_one_line, print_all_properties
from utils.__init__ import setup_environment
from configs.config import Config
from datasets.ResArch_dataset import ResArch_Dataset
from model.ResArch import prepare_model_and_train, train_gt
setup_environment()

# these lines are for reading flags
parser = argparse.ArgumentParser(prog="Speaker Recognition training")
DEFAULT_YAML = './config.yml'
parser.add_argument("--config", "-c", type=str, default=DEFAULT_YAML, help="The file path of model configuration file")
parser.add_argument("--train_type", type=str, default='compile', help="Gradient_tape or compile type?")
parser.add_argument("--model", type=str, default='ResArch', help='Which model Do you want to train on?(SpeakerNet/ResArch/SincNet)')
args = parser.parse_args()


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

                             
  my_ds_object.prepare_tf_dataset_obj()
  print_all_properties(my_ds_object) 
  if args.train_type == 'compile':
     history, model = prepare_model_and_train(my_ds_object.sample_rate,
                                      my_ds_object.class_names,
                                      my_ds_object.train_ds,
                                      my_ds_object.epochs,
                                      my_ds_object.valid_ds)
  elif args.train_type == 'Gradient_tape':
      model = train_gt(my_ds_object.epochs, my_ds_object.sample_rate,
           my_ds_object.class_names, my_ds_object.train_ds)

# read Config file.
if args.model == "ResArch":
    config = Config(args.config)
    print("You selected to use ResArch")
    train_ResArch(config)

elif args.model == "SincNet":
    print("you selected SincNet. NOT IMPLEMENTED YET")
elif args.model == "SpeakerNet":
    print("you selected SpeakerNet. NOT IMPLEMENTED YET")





