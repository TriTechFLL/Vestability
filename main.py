# import system modules
import sys
from struct import unpack
import math
import time

# import dependencies
import sounddevice
# NumPy for fast math
import numpy as np
# Sense HAT controller for LEDs
from sense_hat import SenseHat
# Uncomment this and comment the above to use emulator instead
# from sense_emu import SenseHat

# Initialize Sense HAT
sense = SenseHat()
sense.clear()

# sense = SenseHat()

# Get a list of valid audio input devices from sounddevice
def get_device(name_match, api_name, include_outputs):
    devices = sounddevice.query_devices()
    result = None
    for dev in devices:
        if include_outputs or dev['max_input_channels'] > 0:
            # get additional info about the API used to access the device so we can verify it's the one we want from audio_api
            api = sounddevice.query_hostapis(dev['hostapi'])
            print(f"{dev['index']}. {dev['name']} via {api['name']} ({dev['max_input_channels']}) ({dev['default_samplerate']}kHz)")
            if not result and name_match in dev['name'] and api['name'] == api_name:
                # This is the one we want! Store it in result so we can return it later.
                print(f"Choosing device {dev['index']}")
                result = dev['index']
    return result

# Audio setup
# Number of channels on input device
no_channels = 1
# Input device sample rate
sample_rate = 48000
# Part of the input device name, used to detect it. We use the Easy Effects Source to make use of its equalizer preset.
device_name = "Easy Effects Source"
# System Audio API to use - this will be "JACK Audio Connection Kit" for inclusive capture or "ALSA" for exclusive capture on Linux
audio_api = "JACK Audio Connection Kit"
# If local loopback outputs should be included, enable this to detect local music playback
include_outputs = False
# ms delay between reading data from the microphone
block_duration = 5

# Audio gain for the display - increase this to make it more sensitive
gain = 8000

# Call get_device to get the output device we want
device = get_device(device_name, audio_api, include_outputs)
if not device:
    print("No matching input device found")
    sys.exit(1)

# Colors configuration
rotation = 0
# yellow = (255, 255, 0)
red = (255, 0, 0)
blue = (0, 0, 255)
green = (0, 204, 0)
e = (0, 0, 0)  # Empty
# Initialize an array corresponding to the display pixels
empty = [
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
 e, e, e, e, e, e, e, e,
]

# Distribution
# spectrum = [green, green, green, yellow, yellow, yellow, red, red]
# Used to store the level of each bar so we cna process it into display output
matrix = [0, 0, 0, 0, 0, 0, 0, 0]

# power = []
# Set up weighting for each bar
weighting = [4, 8, 8, 16, 16, 32, 32, 64]
# Multiply each weighting value above by 8
weighting = [x*8 for x in weighting]

def piff(frequency, frames):
    """
    Returns the index in the power array corresponding to a particular frequency
    frequency: the target frequency to find
    frames: audio frame size
    """
    return int(2 * frames * frequency / sample_rate)

def volume_frequency_range(power, frames, freq_low, freq_high):
    """
    Returns integer value for volume calculated between freq_low and freq_high 
    power: amplitude data
    frames: audio frame size
    freq_low, freq_high: frequency range
    """
    try:
        volume = int(np.mean(power[piff(freq_low, frames):piff(freq_high, frames):1]))
        return volume
    except:
        print("Frequency range too small! idx:", piff(freq_low, frames), piff(freq_high, frames))
        return 0

def calculate_levels(data, frames):
    """
    Calculate matrix from the audio data
    data: audio data
    frames: audio frame count
    """
    global matrix

    # Apply FFT - real data
    fourier = np.fft.rfft((data[:, 0]) * gain)
    # Remove last element in array to make it the same size as frame count
    fourier = np.delete(fourier, len(fourier) - 1)
    # Find average amplitude for specific frequency ranges in Hz
    power = np.abs(fourier)
    # first one overlaps a little because if its < 100 range you get errors
    matrix[0] = volume_frequency_range(power, frames, 0, 200)
    matrix[1] = volume_frequency_range(power, frames, 156, 313)
    matrix[2] = volume_frequency_range(power, frames, 313, 625)
    matrix[3] = volume_frequency_range(power, frames, 625, 1250)
    matrix[4] = volume_frequency_range(power, frames, 1250, 2500)
    matrix[5] = volume_frequency_range(power, frames, 2500, 5000)
    matrix[6] = volume_frequency_range(power, frames, 5000, 10000)
    matrix[7] = volume_frequency_range(power, frames, 10000, 20000)

    # Clean up final column values for the LED matrix
    matrix = np.divide(np.multiply(matrix, weighting), 1000000)
    matrix = matrix.clip(0, 24)
    return matrix

def data_callback(data, frames_all_channels, time, status):
    """
    Called when new data is ready from the microphone
    data: audio data
    
    """
    # We only want to process the data from one channel, so divide the frame count by the channel count
    frames = frames_all_channels / no_channels
    # Generate the matrix
    matrix = calculate_levels(data, frames)
    # Copy the empty display array to draw the visualization to the display
    figure = empty[:]
    for y in range(0, 8):
        # loudness level is always 0-24.
        # draw the green bar
        for x in range(0, min(int(matrix[y]), 8)):
            figure[(y) * 8 + x] = green
        # if loudness >= 16, draw a blue bar on the top 4 pixels
        if int(matrix[y]) >= 16:
            for x in range(4):
                figure[(y+1) * 8 - x - 1] = blue
        # if loudness == 24, draw a red bar on the top 2 pixels
        if int(matrix[y]) == 24:
            for x in range(2):
                figure[(y+1) * 8 - x - 1] = red
    # print("rotation: ", rotation)
    # print("figure: ", figure)
    # Figure now contains the display data we just created, set the display's rotation (needs to be done each cycle) and send it to the display!
    sense.set_rotation(rotation)
    sense.set_pixels(figure)

# Create an input streamand start it
stream = sounddevice.InputStream(device = device, channels = no_channels, blocksize=int(sample_rate * block_duration / 1000), samplerate=sample_rate, callback=data_callback)
stream.start()
try:
    while True:
        # sounddevice uses callbacks so the main thread no longer has to do anything, but it needs to still be running
        time.sleep(10)
except KeyboardInterrupt:
    # Handle exit requests and clean up the stream.
    print("Ctrl-C Terminating...")
    stream.stop()
    stream.close()
    sense.clear()
    sys.exit(1)
