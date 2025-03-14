from openai import OpenAI
import os
import tiktoken
import math
import numpy as np
import ffmpeg

client = OpenAI(api_key="")

def num_of_tokens_from_string(text: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    return len(tokens)

def split_audio(input_audio, chunk_length=900, delete_original=True):
    """Splits an MP3 file into 15-minute chunks and deletes the original file after splitting."""
    try:
        # Get audio duration
        probe = ffmpeg.probe(input_audio)
        duration = float(probe['format']['duration'])

        # Calculate number of chunks
        num_chunks = math.ceil(duration / chunk_length)
        output_files = []

        for i in range(num_chunks):
            start_time = i * chunk_length
            output_file = f"chunk_{i}.mp3"
            output_files.append(output_file)

            ffmpeg.input(input_audio, ss=start_time, t=chunk_length).output(
                output_file, format="mp3", acodec="libmp3lame"
            ).run(overwrite_output=True)

        # ✅ Delete the original file after splitting
        if delete_original:
            os.remove(input_audio)
            print(f"Deleted original file: {input_audio}")

        return output_files  # List of chunked MP3 files

    except Exception as e:
        print("Error:", e)
        return []

def get_embedding(text, model="text-embedding-3-large"):
    text = text.replace("\n", " ")
    response = openai.embeddings.create(input=[text], model=model)
    embedding = response.data[0].embedding
    return embedding

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
def split_audio(input_audio, chunk_length=900, delete_original=True):
    probe = ffmpeg.probe(input_audio)
    duration = float(probe['format']['duration'])
    num_chunks = math.ceil(duration / chunk_length)
    output_files = []
    
    for i in range(num_chunks):
        start_time = i * chunk_length
        output_file = f"chunk_{i}.mp3"
        output_files.append(output_file)
        ffmpeg.input(input_audio, ss=start_time, t=chunk_length).output(
            output_file, format="mp3", acodec="libmp3lame"
        ).run(overwrite_output=True)

    if delete_original:
        os.remove(input_audio)
        print(f"Deleted original file: {input_audio}")

    return output_files

# Transcribe audio to get transcript
def transcribe_audio(audio_path):
    with open(audio_path, "rb") as file:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=file
        )
    return transcript.text