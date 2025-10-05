import subprocess

def speak(text, model_path="piper/en_US-ryan-low.onnx", output_path="sample_audio/output.wav"):
    """
    Convert text to speech using Piper (offline).

    Args:
        text (str): Text to synthesize
        model_path (str): Path to Piper voice model (.onnx)
        output_path (str): Path to save output WAV file
    """
    process = subprocess.Popen(
        ["./piper/piper", "--model", model_path, "--output_file", output_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    process.communicate(input=text)
    print(f"Audio saved to {output_path}")

if __name__ == "__main__":
    speak("Hello! This is a test using Piper offline text to speech.")
