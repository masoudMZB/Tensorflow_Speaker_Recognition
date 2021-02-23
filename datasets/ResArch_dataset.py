from datasets.BaseClass import BaseDataset
import tensorflow as tf
import os
from os.path import isfile, join
import numpy as np
import shutil
from tensorflow import keras
from pathlib import Path
from IPython.display import display, Audio
import subprocess



class ResArch_Dataset(BaseDataset):
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
                stage : str,
                shuffle: bool = False,
              ):
        super(ResArch_Dataset, self).__init__(
          dataset_root = dataset_root,
          my_audios_folder = my_audios_folder,
          my_noises_folder = my_noises_folder,
          valid_split = valid_split,
          shuffle_seed = shuffle_seed,
          sample_rate = sample_rate,
          scale = scale,
          batch_size = batch_size,
          epochs = epochs,
          shuffle = shuffle,
          stage = stage
        )
        self.train_ds = None
        self.valid_ds = None
        self.noise_paths = []
        self.noises = []

    def create_tf_dataset_object():
      print('salam')

    # ------------------------------------------------------ download data function ------------------------------

    def download_dataset_from_kaggle(self, dataset_username_owner = 'kongaevans', dataset_name = 'speaker-recognition-dataset', path_to_unzip = './train'):
      # If you want to dwonload a dataset from kaggle don't forger to set your apikey and username
      import os
      # Write your own KAGGLE_USERNAME and KAGGLE_KEY
      os.environ['KAGGLE_USERNAME'] = "masoudmzb"
      os.environ['KAGGLE_KEY'] = "4dae19325a38899d87ed3aac64e46dfe"
      import kaggle
      kaggle.api.authenticate()
      kaggle.api.dataset_download_files( dataset_username_owner+ '/' + dataset_name , path=path_to_unzip, unzip=True)


    # ------------------------------------------  prepare  audios -------------------------

    def prepare_folders_for_kaggle_dataset(self):
      '''
      we need this directory type:
      main_directory/
          ...speaker_a/
          ...speaker_b/
          ...speaker_c/
          ...speaker_d/
          ...speaker_e/
          ...other/
          ..._background_noise_/
          After sorting, we end up with the following structure:
 
      main_directory/
          ...audio/
          ......speaker_a/
          ......speaker_b/
          ......speaker_c/
          ......speaker_d/
          ......speaker_e/
          ...noise/
          ......other/
          ......_background_noise_/
      '''
      # If folder `audio`, does not exist, create it, otherwise do nothing
      if os.path.exists(self.dataset_audios_path) is False:
          os.makedirs(self.dataset_audios_path)

      # If folder `noise`, does not exist, create it, otherwise do nothing
      # MY OWN NOTE : LET's SEE tf.io.gfile
      if tf.io.gfile.exists(self.dataset_noises_path) is False:
          tf.io.gfile.makedirs(self.dataset_noises_path)

      for folder in os.listdir(self.dataset_root):
          if os.path.isdir(os.path.join(self.dataset_root, folder)):
              if folder in [self.my_audios_folder, self.my_noises_folder]:
                  # If folder is `audio` or `noise`, do nothing
                  continue
              elif folder in ["other", "_background_noise_"]:
                  # If folder is one of the folders that contains noise samples,
                  # move it to the `noise` folder
                  shutil.move(
                      os.path.join(self.dataset_root, folder),
                      os.path.join(self.dataset_noises_path, folder),
                  )
              else:
                  # Otherwise, it should be a speaker folder, then move it to
                  # `audio` folder
                  shutil.move(
                      os.path.join(self.dataset_root, folder),
                      os.path.join(self.dataset_audios_path, folder),
                  )


    def get_all_noises_files(self):
      # Get the list of all noise files
        for subdir in tf.io.gfile.listdir(self.dataset_noises_path):
            subdir_path = Path(self.dataset_noises_path) / subdir
            if os.path.isdir(subdir_path):
                self.noise_paths += [
                    os.path.join(subdir_path, filepath)
                    for filepath in os.listdir(subdir_path)
                    if filepath.endswith(".wav")
                ]

        print(
            "Found {} files belonging to {} directories".format(
                len(self.noise_paths), len(os.listdir(self.dataset_noises_path))
            )
        )

    def change_sample_rates(self):
        command = (
      "for dir in `ls -1 " + self.dataset_noises_path + "`; do "
      "for file in `ls -1 " + self.dataset_noises_path + "/$dir/*.wav`; do "
      "sample_rate=`ffprobe -hide_banner -loglevel panic -show_streams "
      "$file | grep sample_rate | cut -f2 -d=`; "
      "if [ $sample_rate -ne 16000 ]; then "
      "ffmpeg -hide_banner -loglevel panic -y "
      "-i $file -ar 16000 temp.wav; "
      "mv temp.wav $file; "
      "fi; done; done"
                )

        os.system(command)

    # Split noise into chunks of 16000 each
    def load_noise_sample(self, path):
        sample, sampling_rate = tf.audio.decode_wav(
        tf.io.read_file(path), desired_channels=1
    )
        if sampling_rate == self.sample_rate:
            # Number of slices of 16000 each that can be generated from the noise sample
            slices = int(sample.shape[0] / self.sample_rate)
            sample = tf.split(sample[: slices * self.sample_rate], slices)
            return sample
        else:
            print("Sampling rate for {} is incorrect. Ignoring it".format(path))
            return None


    def store_all_noises(self):
        for path in self.noise_paths:
            sample = self.load_noise_sample(path)
            if sample:
                self.noises.extend(sample)
        self.noises = tf.stack(self.noises)

        print(
            "{} noise files were split into {} noise samples where each is {} sec. long".format(
                len(self.noise_paths), self.noises.shape[0], self.noises.shape[1] // self.sample_rate
            )
        )
    # --------------------------     create TFDATASET functions   -------------------------------
    def paths_and_labels_to_dataset(self, audio_paths, labels):
        """Constructs a dataset of audios and labels."""
        # giving an array so we can create tf.data.Dataset.from_tensor_slice(list)
        path_ds = tf.data.Dataset.from_tensor_slices(audio_paths)
        audio_ds = path_ds.map(lambda x : self.path_to_audio(x))
        label_ds = tf.data.Dataset.from_tensor_slices(labels)
        return tf.data.Dataset.zip((audio_ds, label_ds))

    def path_to_audio(self,path):
        """Reads and decodes an audio file."""
          # tf.audio.decode_wav(
    #     contents, desired_channels=-1, desired_samples=-1, name=None
    # )
        audio = tf.io.read_file(path)
        audio, _ = tf.audio.decode_wav(audio, 1, self.sample_rate)
        return audio


    def add_noise(self, audio, noises=None, scale=0.5):
        if noises is not None:
            # Create a random tensor of the same size as audio ranging from
            # 0 to the number of noise stream samples that we have.
            tf_rnd = tf.random.uniform(
                (tf.shape(audio)[0],), 0, noises.shape[0], dtype=tf.int32
            )
            # https://www.geeksforgeeks.org/python-tensorflow-gather/ این ایندکس هایی که میگمو از این ارایه به من بده
            noise = tf.gather(noises, tf_rnd, axis=0)

            # Get the amplitude proportion between the audio and the noise
            prop = tf.math.reduce_max(audio, axis=1) / tf.math.reduce_max(noise, axis=1)
            prop = tf.repeat(tf.expand_dims(prop, axis=1), tf.shape(audio)[1], axis=1)

            # Adding the rescaled noise to audio
            audio = audio + noise * prop * scale

        return audio

    def audio_to_fft(self, audio):
        # Since tf.signal.fft applies FFT on the innermost dimension,
        # we need to squeeze the dimensions and then expand them again
        # after FFT
        audio = tf.squeeze(audio, axis=-1)
        # https://blog.faradars.org/complex-numbers/   برای محاسبه فوریه باید ورودی بصورت اعداد مختلط باشن
        fft = tf.signal.fft(
            tf.cast(tf.complex(real=audio, imag=tf.zeros_like(audio)), tf.complex64)
        )
        fft = tf.expand_dims(fft, axis=-1)

        # Return the absolute value of the first half of the FFT
        # which represents the positive frequencies
        return tf.math.abs(fft[:, : (audio.shape[1] // 2), :])

    def prepare_train_df_val_df(self):
        class_names = tf.io.gfile.listdir(self.dataset_audios_path)
        print("Our class names: {}".format(class_names,))

        audio_paths = []
        labels = []

        for label, name in enumerate(class_names):
            print("Processing speaker {}".format(name,))
            dir_path = Path(self.dataset_audios_path) / name
            speaker_sample_paths = [
                os.path.join(dir_path, filepath)
                for filepath in os.listdir(dir_path)
                if filepath.endswith(".wav")
            ]
            audio_paths += speaker_sample_paths
            labels += [label] * len(speaker_sample_paths)

        print(labels)

        print(
            "Found {} files belonging to {} classes.".format(len(audio_paths), len(class_names))
        )

        # Shuffle

        '''
        why does it work? 
        arr = [1,2,3,4,5,6,7,8,9]
        arr2 = [11,22,33,44,55,66,77,88,99]

        rng = np.random.RandomState(self.shuffle_seed)
        rng.shuffle(arr)
        rng = np.random.RandomState(self.shuffle_seed)
        rng.shuffle(arr2)

        arr, arr2
        >> ([4, 9, 7, 8, 3, 6, 2, 1, 5], [44, 99, 77, 88, 33, 66, 22, 11, 55])
        '''
        rng = np.random.RandomState(self.shuffle_seed)
        rng.shuffle(audio_paths)
        rng = np.random.RandomState(self.shuffle_seed)
        rng.shuffle(labels)


        # Split into training and validation
        num_val_samples = int(self.valid_split * len(audio_paths))
        print("Using {} files for training.".format(len(audio_paths) - num_val_samples))
        train_audio_paths = audio_paths[:-num_val_samples]
        train_labels = labels[:-num_val_samples]

        print("Using {} files for validation.".format(num_val_samples))
        valid_audio_paths = audio_paths[-num_val_samples:]
        valid_labels = labels[-num_val_samples:]

        '''
        This dataset fills a buffer with buffer_size elements, then randomly samples elements from this buffer, 
        replacing the selected elements with new elements. For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required.

        For instance, if your dataset contains 10,000 elements but buffer_size is set to 1,000, 
        then shuffle will initially select a random element from only the first 1,000 elements in the buffer.
        Once an element is selected, its space in the buffer is replaced by the next (i.e. 1,001-st) element, maintaining the 1,000 element buffer.

        reshuffle_each_iteration controls whether the shuffle order should be different for each epoch. In TF 1.X, 
        the idiomatic way to create epochs was through the repeat transformation:
        '''

        # Create 2 datasets, one for training and the other for validation
        self.train_ds = self.paths_and_labels_to_dataset(train_audio_paths, train_labels)
        self.train_ds = self.train_ds.shuffle(buffer_size=self.batch_size * 8, seed=self.shuffle_seed).batch(
            self.batch_size
        )

        self.valid_ds = self.paths_and_labels_to_dataset(valid_audio_paths, valid_labels)
        self.valid_ds = self.valid_ds.shuffle(buffer_size=32 * 8, seed=self.shuffle_seed).batch(32)

    def add_noise_and_feature_Extraction(self):
        # Add noise to the training set
        self.train_ds = self.train_ds.map(
            lambda x, y: (self.add_noise(x, self.noises, scale=self.scale), y),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )

        # Transform audio wave to the frequency domain using `audio_to_fft`
        self.train_ds = self.train_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        '''
        Creates a Dataset that prefetches elements from this dataset.

        Most dataset input pipelines should end with a call to prefetch. This allows later elements to be prepared while the current
        element is being processed. This often improves latency and throughput, at the cost of using additional memory to store prefetched elements.

        Note: Like other Dataset methods, prefetch operates on the elements of the input dataset. It has no concept of examples vs.
        batches. examples.prefetch(2) will prefetch two elements (2 examples), while examples.batch(20).prefetch(2) will prefetch 2
          elements (2 batches, of 20 examples each).
        '''


        '''
        The tf.data API provides a software pipelining mechanism through the tf.data.Dataset.prefetch transformation, 
        which can be used to decouple the time when data is produced from the time when data is consumed. In particular,
        the transformation uses a background thread and an internal buffer to prefetch elements from the input dataset ahead of
          the time they are requested. The number of elements to prefetch should be equal to (or possibly greater than) the number of
          batches consumed by a single training step. You could either manually tune this value, or set it 
          to tf.data.experimental.AUTOTUNE which will prompt the tf.data runtime to tune the value dynamically at runtime.
        '''

        self.train_ds = self.train_ds.prefetch(tf.data.experimental.AUTOTUNE)

        self.valid_ds = self.valid_ds.map(
            lambda x, y: (self.audio_to_fft(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        self.valid_ds = self.valid_ds.prefetch(tf.data.experimental.AUTOTUNE)

    def prepare_tf_dataset_obj(self):
        # self.download_dataset_from_kaggle(dataset_username_owner = 'kongaevans', dataset_name='speaker-recognition-dataset')
        # self.prepare_folders_for_kaggle_dataset()
        self.get_all_noises_files()
        self.change_sample_rates()
        self.store_all_noises()
        self.prepare_train_df_val_df()
        self.add_noise_and_feature_Extraction()

    def get_train_ds(self):
      return self.train_ds
    def get_valid_ds(self):
      return self.valid_ds











