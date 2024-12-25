import os
import glob
import sklearn

import mne
from mne_connectivity.viz import plot_sensors_connectivity
from mne.viz import plot_epochs_psd_topomap

from PIL import Image
import pyvista
import pyvistaqt
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

# local modules
from home.data import DataLoader
from home.feature import FeatureExtraction
from home import visual

# objects
data_loader=DataLoader()
features=FeatureExtraction()

# paths
image_save_path = os.path.join('web_app/app/static', 'images')

# function to generate folders for different visualizations
def gen_folder(folder_name):
    folder_path = os.path.join(image_save_path, folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
        
# generate gif of selected images
def make_gif(folder_name,gif_name):
    dir = os.path.join(image_save_path, folder_name)
    gif_dir = os.path.join(image_save_path, gif_name)
    frames = [Image.open(image) for image in glob.glob(f"{dir}/*")]
    frame_one = frames[0]
    frame_one.save(gif_dir, format="GIF", append_images=frames,
               save_all=True, duration=750, loop=2)

def create_visuals(filepath):
    epochs = data_loader.load_file(filepath)
    
    ### power spectral density / psd
    psd_folder = 'psd'
    psd_gif = 'psd.gif'
    gen_folder(psd_folder)
    psd_dir = os.path.join(image_save_path, psd_folder)
    for i in range(10): 
        epochs['1'][i].plot_psd(picks='eeg').savefig(os.path.join(psd_dir,'00%d.png'%i))
    make_gif(psd_folder,psd_gif)
    
    ### psd_topomap
    psd_topo_folder = 'psd_topo'
    psd_topo_gif = 'psd_topo.gif'
    gen_folder(psd_topo_folder)
    psd_topo_dir = os.path.join(image_save_path, psd_topo_folder)
    for i in range(10): 
        plot_epochs_psd_topomap(epochs['1'][i], ch_type='eeg', cmap='RdBu_r', outlines='head').savefig(os.path.join(psd_topo_dir,'00%d.png'%i))
    make_gif(psd_topo_folder,psd_topo_gif)

   ### Independent component analysis 
    ica_folder = 'ica'
    ica_gif = 'ica.gif'
    gen_folder(ica_folder)
    ica_dir = os.path.join(image_save_path, ica_folder)
   # set up and fit the ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter='auto')
    ica.fit(epochs)
    #ica.exclude = [1, 2]  # to display only selected ICAs
    #ica_plots  = []
    ica_plots = ica.plot_properties(epochs) #(data, picks=ica.exclude)
    for i in range(5): 
        ica_plots[i].savefig(os.path.join(ica_dir,'00%d.png'%i))
    make_gif(ica_folder,ica_gif)

    ### Joint Plots
    cov = mne.compute_covariance(epochs, tmax=0.)
    evoked = epochs['1'].average()  # trigger 1 in auditory/left
    evoked.plot_joint().savefig('web_app/app/static/images/joint_plot.png')

    ### connectivity
    con_array = features.connectivity_features(epochs)
    # Do the actual plotting
    connectivity_fig = plot_sensors_connectivity(
        epochs.info,
        con_array[:, :, 0])
    # Save the 3D scene as grapic
    conn_svg_filepath = os.path.join(image_save_path, 'conn.svg')
    connectivity_fig.plotter.save_graphic(conn_svg_filepath, title='Sensors Connectivity', raster=True, painter=True)
    # Save 3D scene to html file
    conn_html_filepath = os.path.join(image_save_path, 'conn.html')
    connectivity_fig.plotter.export_html(conn_html_filepath)
    # Take a screenshot of the 3D scene and save as image
    screenshot = connectivity_fig.plotter.screenshot()
    # The screenshot is just a NumPy array, so we can display it via imshow()
    # and then save it to a file.
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(screenshot, origin='upper')
    ax.set_axis_off()  # Disable axis labels and ticks
    fig.tight_layout()
    conn_2d_filepath = os.path.join(image_save_path, 'conn.png')
    fig.savefig(conn_2d_filepath, dpi=150)