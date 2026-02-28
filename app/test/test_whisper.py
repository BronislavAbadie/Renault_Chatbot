import whisper
import numpy as np
import soundfile as sf
from scipy.signal import resample

def transcribe_audio(file_path):
    """
    Transcribes a .wav file using Whisper without relying on ffmpeg.
    Args:
        file_path (str): Path to the .wav file.
    Returns:
        str: Transcription of the audio.
    """
    # Load the Whisper model
    model = whisper.load_model("base")

    # Load the audio file directly as a NumPy array
    audio, sample_rate = sf.read(file_path)

    # Ensure the audio is mono (convert if necessary)
    if len(audio.shape) > 1:  # Check if the audio has multiple channels
        audio = np.mean(audio, axis=1)  # Convert to mono by averaging channels

    # Resample the audio to 16 kHz (required by Whisper)
    if sample_rate != 16000:
        num_samples = int(len(audio) * 16000 / sample_rate)
        audio = resample(audio, num_samples)

    # Convert the audio to float32 (required by Whisper)
    audio = audio.astype(np.float32)

    # Trim or pad the audio to 30 seconds (required by Whisper)
    max_length = 30 * 16000  # 30 seconds at 16 kHz
    if len(audio) > max_length:
        audio = audio[:max_length]  # Trim to 30 seconds
    else:
        audio = np.pad(audio, (0, max_length - len(audio)))  # Pad with zeros

    # Convert the audio to the format expected by Whisper
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # Transcribe the audio
    options = whisper.DecodingOptions(fp16=False,
                                      task='transcribe',
                                      language='fr')
    result = whisper.decode(model, mel, options)

    # Return the transcription
    return result.text

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python test_whisper.py <path_to_wav_file>")
        sys.exit(1)

    wav_file = sys.argv[1]
    transcription = transcribe_audio(wav_file)
    print("Transcription:")
    print(transcription)