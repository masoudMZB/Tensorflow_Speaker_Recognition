import abc
import os

class BaseDataset(metaclass=abc.ABCMeta):
  def __init__(self, 
              dataset_root: str,
              my_audios_folder : str,
              my_noises_folder : str,
              valid_split: float,
              shuffle_seed : int,
              sample_rate: int,
              scale: float,
              batch_size : int,
              epochs : int,
              shuffle: bool = False,
              stage: str = "train",):
        self.shuffle = shuffle
        self.stage = stage
        self.dataset_root = dataset_root
        self.my_audios_folder = my_audios_folder
        self.my_noises_folder = my_noises_folder
        self.valid_split = valid_split
        self.shuffle_seed = shuffle_seed
        self.sample_rate = sample_rate
        self.scale = scale
        self.batch_size = batch_size
        self.epochs = epochs
        self.dataset_audios_path = os.path.join(self.dataset_root, self.my_audios_folder)
        self.dataset_noises_path = os.path.join(self.dataset_root, self.my_noises_folder)


  @abc.abstractmethod
  def create_tf_dataset_object(self, *args, **kwargs):
    raise NotImplementedError()

    