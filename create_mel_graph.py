import numpy as np
import matplotlib.pyplot as plt
import IPython.display
import librosa
import librosa.display

filename = ["problem.wav", "problem1.wav", "problem2.wav", "problem3.wav", "problem4.wav", "problem5.wav"]

filedirectory = ["problem/sample_Q_E01/",
                 "problem/sample_Q_E02/"]

filename_list = []

for directory in filedirectory:
    for name in filename:
        filename_list.append(directory+name)

print(filename_list)

# For train sound files
for i in range(len(filename_list)):
    try:
        q, q_sr = librosa.load(filename_list[i])
    except:
        continue
    # Let's make and display a mel-scaled power (energy-squared) spectrogram
    S = librosa.feature.melspectrogram(q, sr=q_sr, n_mels=128)

    # Convert to log scale (dB). We'll use the peak power as reference.
    log_S = librosa.power_to_db(S, ref=np.max)

    # Make a new figure
    plt.figure(figsize=(12,4))

    # Display the spectrogram on a mel scale
    # sample rate and hop length parameters are used to render the time axis
    librosa.display.specshow(log_S, sr=q_sr, x_axis='time', y_axis='mel')
    # print(log_S)

    # Put a descriptive title on the plot
    plt.title('mel power spectrogram')

    # draw a color bar
    plt.colorbar(format='%+02.0f dB')

    # Make the figure layout compact
    plt.tight_layout()
    plt.savefig("test_mel_img/E01_E02_E03/"+str(i)+".png")