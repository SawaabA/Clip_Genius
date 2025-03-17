from vector_database import get_transcript, split_text_tokens

video_url = "https://www.youtube.com/watch?v=ua-xR4SFXAs"
transcript = get_transcript(video_url)

if "Error retrieving transcript" not in transcript:
    split_transcript = split_text_tokens(transcript)

    print("\n✅ Transcript Retrieved and Split into Chunks:\n")
    for i, chunk in enumerate(split_transcript):
        print(f"Chunk {i+1}: {chunk}\n")
else:
    print("\n❌ Failed to retrieve transcript:", transcript)