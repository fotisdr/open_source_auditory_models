# -*- coding: utf-8 -*-
"""
This file contains all the supplementary functions needed to execute the
ICNet example script in Python.

@author: Fotios Drakopoulos, UCL, June 2024
"""

from activations import *
from layers import *

from typing import List, Optional, Tuple, Union
from glob import glob
import matplotlib.pyplot as plt

import numpy as np
import scipy.io as sp_io
import scipy.signal as sp_sig

import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import tensorflow_probability as tfp


def rms(x: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Compute RMS energy of a matrix over the specified axis.
    If axis = None then the RMS is computed over all axes.
    """
    sq = np.mean(np.square(x), axis = axis)
    return np.sqrt(sq)

def resample_and_filter(signal: np.ndarray, fs_signal: float, fs_target: float,
                         filter_order: int = 8, axis: int = 0) -> np.ndarray:
    """
    Resample an audio signal to the fs_target sampling frequency.
    If the signal gets downsampled, a low-pass filter is first applied
    to avoid aliasing. 
    Returns the resampled (and filtered) audio signal.
    """
    if fs_target < fs_signal: # downsampling
        sos = sp_sig.butter(filter_order, 0.99*(fs_target/2), btype='low', analog=False, 
                            fs=fs_signal, output='sos') # low-pass filtering
        # Apply the low-pass digital filter
        signal = sp_sig.sosfiltfilt(sos, signal, axis=axis)
    # Resample the signal
    if fs_target != fs_signal: 
        signal = sp_sig.resample_poly(signal, int(fs_target), fs_signal, axis=axis)

    return signal

def wavfile_read(wavfile: str, fs: float = None) -> Tuple[np.ndarray, float]:
    """
    Read a wavfile and normalize it to +/-1.
    If fs is given, the signal is resampled to the given sampling frequency.
    Returns the sound signal and the corresponding sampling frequency.
    """
    fs_signal, signal = sp_io.wavfile.read(wavfile)
    if not fs:
        fs=fs_signal

    if signal.dtype != 'float32' and signal.dtype != 'float64':
        if signal.dtype == 'int16':
            nb_bits = 16 # -> 16-bit wav files
        elif signal.dtype == 'int32':
            nb_bits = 32 # -> 32-bit wav files
        max_nb_bit = float(2 ** (nb_bits - 1))
        signal = signal / (max_nb_bit + 1.0) # scale the signal to [-1.0,1.0]

    if fs_signal != fs :
        signalr = resample_and_filter(signal, fs_signal, fs)
    else:
        signalr = signal

    return signalr, fs

def pad_along_1dimension(data: np.ndarray, npad_left: int = 0, npad_right: int = 0, 
                         axis: int = 0) -> np.ndarray:
    """
    Pads data with zeros on both sides along the dimension defined.
    """
    # define the npad tuple
    npad = [(0, 0)] * data.ndim
    npad[axis] = (npad_left,npad_right)
    # pad across the defined axis
    data = np.pad(data, npad, mode='constant', constant_values=0)
    
    return data

def slice_1dsignal(signal: np.ndarray, window_size: int, winshift: int, minlength: int = 0, 
                   left_context: int = 2048, right_context: int = 0) -> np.ndarray:
    """Return windows of the given signal by sweeping in stride fractions of window. Slices
    that are less than minlength are omitted. Input signal must be a 1D-shaped array.

    Args:
      signal: A one-dimensional input waveform
      window_size: The size of each window of data from the signal. 
        If window_size = 0 then the window size is matched to the audio size.
      winshift: How much to shift the window as we progress down the signal. 
        If winshift = window_size then the overlap between windows is 0.
      minlength: Drop (final) windows that have less than this number of samples.
      left_context: How much context to add (from earlier parts of the signal) before
        the current window. (Or add zeros if not enough signal)
      right_context: Like left, but to the right of the current window.
    
    Returns:
      A 3D tensor of size [num_frames x window_size x 1]
    """
    assert len(signal.shape) == 1, "signal must be a 1D-shaped array"

    # concatenate zeros to beginning for adding context
    n_samples = signal.shape[0]
    num_slices = (n_samples)
    slices = [] # initialize empty array 
    
    if window_size == 0:
        window_size = n_samples
        winshift = window_size

    for beg_i in range(0, n_samples, winshift):
        beg_i_context = beg_i - left_context
        end_i = beg_i + window_size + right_context
        if n_samples - beg_i < minlength :
            break
        if beg_i_context < 0 and end_i <= n_samples:
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal[:end_i]])), axis=1)
        elif end_i <= n_samples: # beg_i_context >= 0
            slice_ = np.array([signal[beg_i_context:end_i]])
        elif beg_i_context < 0: # end_i > n_samples
            slice_ = np.concatenate((np.zeros((1, left_context - beg_i)),np.array([signal]), np.zeros((1, end_i - n_samples))), axis=1)
        else :
            slice_ = np.concatenate((np.array([signal[beg_i_context:]]), np.zeros((1, end_i - n_samples))), axis=1)

        slices.append(slice_)
    slices = np.vstack(slices)
    slices = np.expand_dims(slices, axis=2) # the CNN will need 3D data
    
    return slices

def get_sorted_channels(channel_CFs: np.ndarray, n_units: int = 10, 
                        CF_min: int = None, CF_max: int = None,
                        plot: bool = False) -> np.ndarray:
    """
    Get the corresponding channel indices that achieve a uniform
    logarithmic spacing based on their CFs (n_units between CF_min 
    and CF_max). Returns the selected channel indices.
    
    Args:
      channel_CFs: CFs of all channels
      n_units: Number of channels to be chosen
      CF_min: Minimum CF to be chosen. If None then the minimum of
        channel_CFs is used.
      CF_max: Maximum CF to be chosen. If None then the maximum of
        channel_CFs is used.
    """    
    # Sort the channel CFs
    sort_indices = np.argsort(channel_CFs)
    
    # Define the frequency range
    if CF_min == None:
        CF_min = channel_CFs.min()
    if CF_max == None:
        CF_max = channel_CFs.max()
    # Frequencies are chosen based on a logarithmic space
    freqs_to_find = np.logspace(np.log10(CF_min), np.log10(CF_max), num=n_units, base=10)
    
    # Match the tone frequencies to the corresponding CFs
    sorted_channel_freqs = []
    sorted_channel_indices = []
    available_CFs = channel_CFs[sort_indices]
    for i, freq in enumerate(freqs_to_find):
        # Find the CFs closest to the chosen frequencies
        freq_diff = np.abs(np.array(available_CFs - freq)) # frequency differences
        fnos = np.where(freq_diff == np.min(freq_diff))[0] # find all channels with closest frequencies
        fno = np.random.choice(fnos) # pick a random channel from those with the lowest difference
        # Append the chosen index and remove from the available CFs
        sorted_channel_indices.append(int(fno))
        available_CFs[int(fno)] = -np.inf
        sorted_channel_freqs.append(channel_CFs[sort_indices][int(fno)])
    # Return the channel indices 
    channel_indices = sort_indices[sorted_channel_indices]
    # Plot the desired logarithmic spacing vs the derived CFs
    if plot:
        plt.semilogy(freqs_to_find/1000,'k--',label='logarithmic spacing')
        plt.plot(channel_CFs[channel_indices]/1000,label='derived CFs')
        plt.xlabel('Units')
        plt.ylabel('Frequency (kHz)')
        plt.legend(frameon=False)

    return channel_indices

def simulate_model_responses(model_path: str, audio_input: np.ndarray, output_to_simulate: str, 
                             time_input: float = 2., print_summary: bool = False, 
                             channel_CFs: np.ndarray = [], CF_min: float = 300., CF_max: float = 12000., 
                             context_size: int = 2048, fs_audio: float = 24414.0625, 
                             fs_MUA: float = 762.939453125, ds_factor: int = 32, p0: float = 2e-5, 
                             audio_normalisation: float = 1/25, time_normalisation: float = 1/36000, 
                             decoder_channels: int = 512, n_decoders: int = 9, 
                             bottleneck_channels: int = 64, n_classes: int = 5, 
                             ) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Function to simulate the responses of the DNN model of the inferior colliculus (ICNet). 
    ICNet was trained to simulate the spiking probabilities across n_classes (0 to 4 spikes).
    The Tensorflow Probability toolbox is used to sample from the predicted distributions and 
    derive multi-unit activity (MUA) across all recorded units from 9 animals.
    
    Args:
      model_path: Where to find the model and weight files
      audio_input: The audio signal that will be used as input
      output_to_simulate: A string that defines the neural response that will be returned from
        the function. Supported choices are: 'units_N','animal_random','animal_X','bottleneck'.
      time_input: The recording time to be used for simulating neural activity (in hours). 
        Values between 0 and 11 should be used, with earlier times corresponding to less
        non-stationarity in the recording. 
      print_summary: Print the DNN model summary
      channel_CFs: The characteristic frequencies (CFs) of all recorded units (channels) in ICNet. 
        If provided, the channel CFs are used to sort the ICNet responses by frequency. When 
        output_to_simulate = 'units_X', X units are selected based on a logarithmic spacing of CF.
      CF_min: The minimum CF to be used for choosing units with logarithmic spacing.
      CF_max: The maximum CF to be used for choosing units with logarithmic spacing.
      context_size, fs_audio, fs_MUA, ds_factor, p0, audio_normalisation, time_normalisation, 
        decoder_channels, n_decoders, bottleneck_channels, n_classes: Model parameters fixed to
        the specific ICNet model version. See the corresponding config.yaml file for more details.

    Returns:
      MUA_response: The simulated ICNet response, according to the provided arguments.
      MUA_CFs: The CFs of the corresponding units in MUA_response. If channel_CFs is not 
        provided, the function returns an empty array (all zeros).
    '''
    ## Define the input/output parameters
    n_batches = audio_input.shape[0]
    window_size_MUA = int((audio_input.shape[1]-context_size) / 32) # after cropping context
    ## Recalibrate the audio input for the model
    audio_input *= audio_normalisation
    ## Load the DNN model
    model_file = model_path + "/model_weights.hdf5"
    model = load_model(model_file, custom_objects={}, compile=False)  
    
    ## Define the model description and output shape
    if 'bottleneck' in output_to_simulate:
        # Get the bottleneck response
        model = Model(model.input[0],model.get_layer('cropped_output').output,name=model.name)
        # Initialise the model output
        n_branches = 1
        n_channels = bottleneck_channels # 64
        output_shape = (n_branches, n_batches, window_size_MUA, n_channels)
        MUA_response = np.zeros(output_shape,dtype=audio_input.dtype)
        MUA_CFs = np.zeros(MUA_response.shape)
    elif 'animal_' in output_to_simulate:
        # Get the response of one animal
        if 'animal_random' in output_to_simulate:
            branch_index = np.random.randint(n_decoders) # random branch (0 to 8)
            print('Animal %d selected' % (branch_index+1))
        else:
            branch_index = int(output_to_simulate.split('_')[-1]) - 1 # use the given index from 1 to 9
        # Redefine the loaded model to only simulate the output that corresponds to the chosen branch
        model_branch_index = branch_index - n_decoders # i - 9
        # Remove the extra time inputs from the model definition
        model_input = [model.input[0],model.input[1][branch_index]]
        model = Model(model_input, model.layers[model_branch_index].output)
        # Initialise the model output
        n_branches = 1
        n_channels = decoder_channels # 512
        output_shape = (n_branches, n_batches, window_size_MUA, n_channels, n_classes)
        MUA_response = np.zeros(output_shape,dtype=audio_input.dtype)
        MUA_CFs = np.zeros(MUA_response.shape[:-1])
    elif 'units_' in output_to_simulate: 
        # Simulate units from all branches (animals)
        n_branches = n_decoders # 9 animals
        n_channels = decoder_channels # 512
        output_shape = (n_branches, n_batches, window_size_MUA, n_channels, n_classes)
        MUA_response = np.zeros(output_shape,dtype=audio_input.dtype)
        MUA_CFs = np.zeros(MUA_response.shape[:-1])
    else:
        raise ValueError('Output choice not supported. Check the documentation for all available options.')
    
    if print_summary:
        model.summary()

    if 'bottleneck' in output_to_simulate:
        ## Simulate the model responses without time 
        for i_batch in range(n_batches):
            stim_input = tf.convert_to_tensor(audio_input[i_batch:i_batch+1])
            MUA_response[:, i_batch:i_batch+1] = model.predict(stim_input, verbose=0)
    else:
        ## Prepare the time input that is needed for the model
        # Starting time is defined by the provided time index
        time_start = np.array(time_input * time_normalisation, dtype=audio_input.dtype)
        # Make a time array from 0 to the end of the frame (N/fs)
        time_array = np.arange(0., window_size_MUA/fs_MUA, 1./fs_MUA, dtype=audio_input.dtype) * time_normalisation
        time_array = np.expand_dims(time_array, axis=(0,-1)) # 3D array
        # The time input will be the generated time array shifted by the measurement time
        time_array += time_start
        # Compute the time shift that will be added to the time array after each batch (for multiple batches)
        time_shift = window_size_MUA / fs_MUA * time_normalisation
    
        ## Simulate the model responses
        for i_batch in range(n_batches):
            stim_input = tf.convert_to_tensor(audio_input[i_batch:i_batch+1])
            time_input = tf.convert_to_tensor(time_array)
            if isinstance(model.input[1], (list,tuple)): # for multiple branches
                time_input = [time_input] * len(model.input[1])

            MUA_response[:, i_batch:i_batch+1] = model.predict((stim_input,time_input), verbose=0)
            # Shift the time array by the time length of the batch
            time_array += time_shift 
    
        ## Sample from the simulated distribution
        # Define the distribution
        distr = tfp.distributions.Categorical(probs=MUA_response, validate_args=False)
        # Sample to get multi-unit neural activity
        MUA_response = distr.sample().numpy().astype(np.int8) # int8 is used for discrete values (0-4)
        
        ## Select and sort the corresponding units (if needed)
        if 'animal_' in output_to_simulate and len(channel_CFs):
            # Sort units based on the CFs of the corresponding branch
            channel_indices = np.argsort(channel_CFs[branch_index*decoder_channels:(branch_index+1)*decoder_channels])
            MUA_response = MUA_response[...,channel_indices]
            # Get the CFs of the corresponding MUA units
            MUA_CFs = channel_CFs[branch_index*decoder_channels:(branch_index+1)*decoder_channels][channel_indices]
        elif 'units_' in output_to_simulate:
            n_units = int(output_to_simulate.split('_')[-1]) # number of units to keep
            if len(channel_CFs):
                # Select units based on a logarithmic spacing of CFs
                channel_indices = get_sorted_channels(channel_CFs, n_units, CF_min, CF_max, plot=False)
            else:
                print('The unit CFs were not provided. Random units will be chosen.')
                channel_indices = np.random.randint(n_decoders*decoder_channels, size=n_units)
            
            MUA_response = np.transpose(MUA_response, (1,2,0,3)) # (n_batches, window_size_MUA, n_branches, n_channels)
            MUA_response = np.reshape(MUA_response, (n_batches, window_size_MUA, -1)) # flatten the two unit dimensions
            # Sort the units based on the chosen indices
            MUA_response = MUA_response[...,channel_indices]
            # Get the CFs of the corresponding MUA units
            MUA_CFs = channel_CFs[channel_indices]

    return MUA_response, MUA_CFs
