import subprocess
import os

def transcribe(audio_path, model_path="whisper.cpp/models/ggml-base.en.bin", translate=False):
    """
    Transcribe or translate speech from an audio file using Whisper.cpp.

    Args:
        audio_path (str): Path to input audio file (.wav, .mp3)
        model_path (str): Path to Whisper model
        translate (bool): If True, translates non-English speech to English
    """
    mode = "--translate" if translate else "--transcribe"
    cmd = [
        "./whisper.cpp/main",
        "-m", model_path,
        "-f", audio_path,
        "--output-txt",
        mode
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output_txt = audio_path.rsplit('.', 1)[0] + ".txt"

    if os.path.exists(output_txt):
        with open(output_txt, "r") as f:
            return f.read().strip()
    else:
        return result.stdout or result.stderr


if __name__ == "__main__":
    text = transcribe("sample_audio/test.wav")
    print("\n Transcription:\n", text)