import tensorflow as tf 
from tensorflow import keras
import numpy as np
import mne
from mne_connectivity import envelope_correlation, spectral_connectivity_epochs
import warnings

# Inside an Epochs object, the data are stored in an array of shape (n_epochs, n_channels, n_times)

FREQ_BANDS = {
    "delta": [0.5, 4.5],
    "theta": [4.5, 8.5],
    "alpha": [8.5, 11.5],
    "sigma": [11.5, 15.5],
    "beta": [15.5, 30]
}

class FeatureExtraction():

    def __init__(self, ch_type:str='eeg', cmap:str='RdBu_r', outlines:str='head'):
        """Initializes FeatureExtraction object.
        Parameters
        ----------
        ch_type : str, default=eeg
            The channel type to plot the figure.
        cmap: str, default='RdBu_r'
            Colormap to use. 
        outlines: str, default='head'
            The outlines to be drawn.
            If 'head', the default head scheme will be drawn.
            If 'skirt' the head scheme will be drawn, but sensors are allowed to be plotted outside of the head circle. 
        Returns
        -------
        None
        """

        self.ch_type = ch_type
        self.cmap = cmap
        self.outlines=outlines
        self.method = None
    
    def transform(self, method, data, labels):
        """
        """
        
        self.method = method
        
        processed_data = []
        processed_labels = []
        
        for _i in range(len(data)):
            sample = data[_i]
            sample_label = labels[_i]
            
            if method == 'raw':
                sample_data = sample.get_data()
                print(sample_data.shape)
                processed_data += sample_data.reshape(sample_data.shape[0], -1).tolist()
                processed_labels += [sample_label]*sample_data.shape[0]
            
            elif method == 'psd':
                psds, freqs = mne.time_frequency.psd_welch(sample, fmin=0.5, fmax=30.)
                # sample_data = np.empty((psds.shape[0], psds.shape[1], 5))
                sample_data = psds
                # for idx, (fmin, fmax) in enumerate(FREQ_BANDS.values()):
                #     psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
                #     sample_data[:,:,idx] = psds_band
                print(sample_data.shape)
                processed_data += sample_data.reshape(sample_data.shape[0], -1).tolist()
                processed_labels += [sample_label]*sample_data.shape[0]
            
            elif method == 'connectivity':
                conn = envelope_correlation(sample)
                sample_data = conn.get_data()
                print(sample_data.shape)
                processed_data += sample_data.reshape(sample_data.shape[0], -1).tolist()
                processed_labels += [sample_label]*sample_data.shape[0]

        
        return np.array(processed_data), np.array(processed_labels)


    def frequency_features(self, epochs):
        """returns mne psd_features as numpy array
        Parameters
        ----------
        epochs : mne Epochs object
            mne.Epochs objects.
        Returns
        -------
        psd_features : mne psd_features as numpy array 
        to view distinct values of the features array --> print(np.where(psd_features != 255)) 
        """
        # save psd image to given path
        psd_topomap = mne.viz.plot_epochs_psd_topomap(epochs, ch_type=self.ch_type, cmap=self.cmap, outlines=self.outlines, show=False).savefig('web_app/app/static/psd_topomap.png')
        
        # load the image via Keras load_img function then convert image into numpy array
        # img = keras.utils.load_img('web_app/app/static/psd_topomap.png')
        # psd_features = tf.keras.utils.img_to_array(img)
        psd_features = mne.time_frequency.tfr_morlet(epochs, freqs = np.arange(7, 30, 3), n_cycles=2, return_itc=False)
        return psd_features        


    def connectivity_features(self, epochs, fmin:float=4., fmax:float=9., method:str='pli', mode:str='multitaper'):
        """returns connectivity matrix as a 2D array
        Parameters
        ----------
        epochs : mne Epochs object
            mne.Epochs objects.
        fmin : float, default=4.
            The lower frequency of interest
        fmax : float, default=9.
            The upper frequency of interest
        method : str, default='pli'
            Connectivity measure(s) to compute
            These can be ['coh', 'cohy', 'imcoh', 'plv', 'ciplv', 'ppc', 'pli', 'wpli', 'wpli2_debiased']
        mode : str, default='multitaper'
            Spectrum estimation mode 
            These can be either: 'multitaper', 'fourier', or 'cwt_morlet'
        Returns
        -------
        conn_features : connectivity matrix as a 2D array 
        """
        # compute connectivity for band containing the evoked response
        info = epochs.info
        sfreq = info['sfreq']  # the sampling frequency
        tmin = 0.0  # exclude the baseline period
        epochs.load_data().pick_types(eeg=True) 
        con = spectral_connectivity_epochs(
            epochs, method=method, mode=mode, sfreq=sfreq, fmin=fmin, fmax=fmax,
            faverage=True, tmin=tmin, mt_adaptive=False, n_jobs=1)

        # get connectivity data as a numpy array
        # ‘dense’ will return each connectivity matrix as a 2D array
        conn_features = con.get_data(output='dense')
        return conn_features
        