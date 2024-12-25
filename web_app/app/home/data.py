
import os
import numpy as np
import pandas as pd
import mne
import time
from mne.datasets import sample
from mne.io import read_raw_fif
from .lsl import LSLClient, MockLSLStream
from werkzeug.utils import secure_filename

class DataLoader():

    def __init__(self, epoch_length:float=1, montage:str='standard_1020'):
        """Initializes DataLoader object.

        Parameters
        ----------
        epoch_length : float, default=1
            Length of one epoch in seconds.

        montage: str, default='standard_1020'
            Name of montage to use. Use `get_builtin_montages()` to
            get a list of supported montages.

        Returns
        -------
        None
        """
        self.epoch_length = epoch_length
        self.montage = montage
        self.data = []
        self.sample_ids = []

    def set_epoch_length(self, epoch_length):
        """Sets `epoch_length` to given value. This determines the
        length on one epoch.

        Parameters
        ----------
        epoch_length : float, default=1
            Length of one epoch in seconds.

        Returns
        -------
        None
        """
        self.epoch_length = epoch_length


    def set_montage(self, montage):
        """Sets `montage` to given value. This determines the EEG channel 
        locations to be used for visualization.

        Parameters
        ----------
        montage: str, default='standard_1020'
            Name of montage to use. Use `get_builtin_montages()` to
            get a list of supported montages.

        Returns
        -------
        None
        """
        self.montage = montage


    def get_builtin_montages(self, descriptions=False):
        """Returns a list of supported montages.

        Parameters
        ----------
        descriptions : bool, default=False
            If `False` a list of names of supported montages is returned.
            If `True` a list of tuples containing montage names and their
            descriptions is returned.

        Returns
        -------
        supported_montages : array-like
            List of supported montages.
        """
        supported_montages = mne.channel.get_builtin_montages(descriptions=descriptions)
        return supported_montages


    def load_file(self, file:str):
        """Loads data from given file as mne.Epochs object.

        Parameters
        ----------
        file : str
            Path to the file to be loaded.

        Returns
        -------
        file_data : mne.Epochs
            Data from the file as mne.Epochs object.
        """

        # Identify format of EEG data
        file_type = file.rsplit('.', 1)[-1]
        
        # Load data 
        try:
            file_data = mne.io.read_raw(file)
        except: #TODO: handle other exceptions
            raise ValueError("Unsupported file type:", file_type, "(from", file,")")

        # Initialize montages if not provided. These give 3d coordinates to the EEG channels for visualization
        if file_data.get_montage() == None:

            montage = mne.channels.make_standard_montage(self.montage)

            # Exclude channels for which montage position is not known
            file_data.drop_channels([channel for channel in file_data.ch_names if channel not in montage.ch_names])

            # Set montages
            file_data.set_montage(montage)

        # Split data to Epoch
        # sfreq = file_data.info['sfreq']     # Sampling frequency
        # np_data = file_data.get_data()      # raw data in numpy array
        # n_timepoints = np_data.shape[-1]    # number of total timepoints in EEG recording
        
        # events = np.array([[i*int(sfreq*self.epoch_length),0,1] for i in range(n_timepoints//int(sfreq*self.epoch_length))])
        # file_data = mne.Epochs(file_data, events, tmin=0, tmax=self.epoch_length, baseline=(0, 0))
        file_data = mne.make_fixed_length_epochs(file_data, duration=self.epoch_length, preload=True)

        return file_data


    def load_folder(self, folder:str):
        """Loads data from given folder as list of mne.Epochs objects.

        Parameters
        ----------
        folder : str, default=1
            Path to folder from which data will be loaded.

        Returns
        -------
        data : array-like of shape (n_files)
            A list of  mne.Epochs objects where each mne.Epochs
            contains data of one file in the folder.
        """

        self.data = []
        self.sample_ids = []

        for file in os.listdir(folder):
            file_data = self.load_file(os.path.join(folder, file))
            self.sample_ids.append(file.split(".")[0])
            self.data.append(file_data)

        return self.data

    def load_target(self, filename):
        """Load
        """
        labels = []
        target_df = pd.read_csv(filename)
        target_df['ID'] = target_df['ID'].apply(secure_filename)

        for _id in self.sample_ids:
            labels.append(target_df[target_df['ID']==_id]['Label'].values[0])

        return labels

    def get_stream_data(self, host=None, epoch_length=1, wait_max=5):
        """Connects to LSL device and returns live streaming data as a mne.Epochs object.

        Parameters
        ----------
        host : str, default=1
            Name of the LSL device to connect to.

        epoch_length: int, default=1
            Length of one epoch of data, in seconds.

        wait_max: int, default=5
            Maximum time to wait for data to be available, in seconds.

        Returns
        -------
        data : mne.Epochs
            A mne.Epochs object containing one epoch of data 
            from LSL device.
        """
        
        with LSLClient(host=host, wait_max=wait_max) as client:
            n_samples = int(epoch_length * client.get_measurement_info()['sfreq'])
            time.sleep(1)
            epoch = client.get_data_as_epoch(n_samples=n_samples)

        return epoch

    def get_mock_stream_data(self, i, epoch_length=1, wait_max=5):
        raw_fname = sample.data_path()  / 'MEG' / 'sample' / 'sample_audvis_filt-0-40_raw.fif'
        raw = read_raw_fif(raw_fname).crop(0, 30).load_data().pick('eeg')
        # raw = mne.io.read_raw("C:/Users/mvallayi/eeg-software/web_app/app/static/uploads/Noel_R_Pre-Deci.bdf").load_data().pick('eeg')
        host = 'mne_stream'
        epoch = None
        with MockLSLStream(host, raw, 'eeg'):
            with LSLClient(info=raw.info, host=host, wait_max=wait_max) as client:
                n_samples = int(client.get_measurement_info()['sfreq'])
                for ii in range(1):
                    epoch = client.get_data_as_epoch(n_samples=n_samples*i)
                # time.sleep(1)
                    print("$$$Epoch Updated")

        return epoch