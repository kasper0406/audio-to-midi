import jax
import jax.numpy as jnp
from numpy.typing import NDArray

from pydub import AudioSegment

import matplotlib.pyplot as plt

def load_audio_and_normalize(file: str) -> (int, NDArray[jnp.float32]):
    """Loads an audio file and returns the sample rate along with the normalized samples.
    """
    SAMPLE_RATE = 44100.0
    audio = AudioSegment.from_file(file, "aac")
    resampled_audio = audio.set_frame_rate(SAMPLE_RATE) # Resample to the frequency we operate in

    left_channel_samples = resampled_audio.split_to_mono()[0].get_array_of_samples()
    sample_array = jnp.array(left_channel_samples).T.astype(jnp.float32)
    normalized_samples = sample_array / max(jnp.max(sample_array), -jnp.min(sample_array))

    return SAMPLE_RATE, normalized_samples

def next_power_of_2(x):  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def fft_audio(sample_rate: int, samples: NDArray[jnp.float32], fft_duration=40) -> (int, NDArray[jnp.float32]):
    """Computes the fft of the audio samples
    Returns a tuple of the frame duration in seconds, and the complex FFT components
    
    Args:
        fft_duration: The duration in ms to compute the fft for. It will be rounded to the next
                      power of 2 samples
    """
    samples_per_fft = next_power_of_2(int(sample_rate * (fft_duration / 1000)))

    num_padding_symbols = samples_per_fft - (samples.shape[0] % samples_per_fft)
    if num_padding_symbols == samples_per_fft:
        num_padding_symbols = 0
    padded_data = jnp.pad(samples, (0, num_padding_symbols), constant_values=0)
    data = padded_data.reshape((int(padded_data.shape[0] / samples_per_fft), samples_per_fft))

    fft = jnp.fft.fft(data)
    #features = jnp.reshape(jnp.stack((fft.real, fft.imag), axis=2), (fft.shape[0], 2 * fft.shape[1]))
    #return features
    return (samples_per_fft / sample_rate, fft)

def cleanup_fft_and_low_pass(duration_per_frame, frames, high_freq_cutoff=10000):
    # Drop phase information and drop high-frequencies above the `high_freq_cut_off` wavelength
    transposed_reals = jnp.transpose(jnp.real(frames))
    frequencies_to_include = int(high_freq_cutoff * duration_per_frame)
    processed_frequencies = transposed_reals[0:frequencies_to_include, :]
    return duration_per_frame, processed_frequencies

def plot_time_domain_audio(sample_rate: int, samples: NDArray[jnp.float32]):
    time_indices = jnp.linspace(0, float(samples.size) / float(sample_rate), samples.size)

    fig, ax = plt.subplots()
    ax.plot(time_indices, samples)

    ax.set(xlabel='time (s)', ylabel='amplitude',
        title='Normalized audio signal in time-domain')
    ax.grid()

def plot_frequency_domain_audio(duration_per_frame: float, frames: NDArray[jnp.float32]):
    fig, ax = plt.subplots()

    # TODO(knielsen): Extract this so it can be used for training, as it is probably better signal?
    #transposed_reals = jnp.transpose(jnp.real(frames))
    #transformed_data = transposed_reals[0:int(transposed_reals.shape[0] / 2), :]
    X = jnp.linspace(0., duration_per_frame * frames.shape[1], frames.shape[1])
    Y = jnp.linspace(0., frames.shape[0] / duration_per_frame, frames.shape[0])
    ax.pcolor(X, Y, frames)

    ax.set(xlabel='Time [s]', ylabel='Frequency [Hz]',
        title='Audio signal in frequency-domain')
    ax.grid()

# sample_rate, samples = load_audio_and_normalize("/Volumes/git/ml/datasets/midi-to-sound/v0/piano_YamahaC7_68.aac")
sample_rate, samples = load_audio_and_normalize("/Volumes/git/ml/datasets/midi-to-sound/v0/piano_YamahaC7_108.aac")
duration_per_frame, frequency_domain = fft_audio(sample_rate, samples)
duration_per_frame, frames = cleanup_fft_and_low_pass(duration_per_frame, frequency_domain)

plot_time_domain_audio(sample_rate, samples)
plot_frequency_domain_audio(duration_per_frame, frames)

plt.show()
