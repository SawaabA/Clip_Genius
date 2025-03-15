from openai import OpenAI
import os
import tiktoken
import math
import numpy as np
import ffmpeg
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

client = OpenAI(api_key="")

def num_of_tokens_from_string(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def cosine_simularity(vector1, vector2):
    vector1 = np.array(vector1)
    vector2 = np.array(vector2)
    dot_product = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Extract an mp3 of the audio from user's mp4 file
def extract_audio(video_path, audio_output="temp_audio.mp3"):
    ffmpeg.input(video_path).output(audio_output, format="mp3", acodec="libmp3lame").run(overwrite_output=True)
    return audio_output

# Split audio into 15 minute segments 
def split_audio(input_audio, output_folder="clips", chunk_length=30, buffer=5, delete_original=True):
    try:
        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)

        probe = ffmpeg.probe(input_audio)
        duration = float(probe['format']['duration'])

        num_chunks = math.ceil(duration / chunk_length)
        output_files = []

        for i in range(num_chunks):
            start_time = i * chunk_length
            actual_chunk_length = min(chunk_length + buffer, duration - start_time)
            output_file = os.path.join(output_folder, f"chunk_{i}.mp3")
            output_files.append(output_file)

            ffmpeg.input(input_audio, ss=start_time, t=actual_chunk_length).output(
                output_file, format="mp3", acodec="libmp3lame"
            ).run(overwrite_output=True)

        # Delete the original file after splitting
        if delete_original:
            os.remove(input_audio)
            print(f"Deleted original file: {input_audio}")

        return output_files

    except Exception as e:
        print("Error:", e)
        return []

# Transcribe audio to get transcript
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )
    return transcript.text


def transcribe_audio_nemo(folder_path="clips/"):
    """Transcribes all MP3 files in the given folder using NeMo and stores timestamps."""
    asr_model = nemo_asr.models.ASRModel.from_pretrained("stt_en_fastconformer_transducer_large")

    transcription_dict = {}  # Store transcriptions with timestamps

    # Ensure we process files in order
    clip_files = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".mp3")], 
        key=lambda x: int(x.split("_")[1].split(".")[0])
    )

    for file in clip_files:
        mp3_path = os.path.join(folder_path, file)
        wav_path = mp3_path.replace(".mp3", ".wav")  # Convert to WAV

        # Convert MP3 to WAV for better accuracy
        ffmpeg.input(mp3_path).output(wav_path, format="wav", ar=16000, ac=1).run(overwrite_output=True)

        # Transcribe WAV file with timestamps
        hypotheses = asr_model.transcribe([wav_path], return_hypotheses=True)

        if isinstance(hypotheses, tuple):  # Extract best hypothesis
            hypotheses = hypotheses[0]

        if not hypotheses or not hasattr(hypotheses[0], 'timestep'):
            print(f"Skipping {file} - No timestamp data available")
            continue

        # Extract timestamps properly
        time_stride = 8 * asr_model.cfg.preprocessor.window_stride
        word_timestamps = hypotheses[0].timestep.get('word', [])

        for stamp in word_timestamps:
            start = stamp['start_offset'] * time_stride
            end = stamp['end_offset'] * time_stride
            word = stamp.get('word', stamp.get('char', ''))
            
            transcription_dict[start] = {"end_time": end, "word": word}

        print(f"✅ Transcribed {file}: {len(word_timestamps)} words")

    return transcription_dict

# Get embedding 
def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding