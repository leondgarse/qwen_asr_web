import os
import sys
import time
import json
import argparse
import subprocess
import requests

def extract_audio(video_path, audio_path):
    print(f"[*] Extracting audio from '{video_path}' to '{audio_path}'...")
    # Extract to a 16kHz mono WAV file which is optimal for ASR models.
    # We use `-loglevel error` to keep the console clean but show errors.
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error",
        audio_path
    ]
    try:
        subprocess.run(cmd, check=True)
        print("[*] Audio extraction complete.")
    except subprocess.CalledProcessError as e:
        print(f"[!] FFmpeg failed during audio extraction: {e}")
        sys.exit(1)

def wait_for_server(url="http://localhost:8000/health"):
    print("[*] Waiting for the server model to load and become ready...")
    while True:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                status = data.get("status")
                if status == "ready":
                    print("[*] Server is ready!")
                    break
                else:
                    print(f"[-] Server status: {status}... waiting.")
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(5)

def transcribe_audio(audio_path, output_path, server_url="http://localhost:8000/transcribe"):
    print(f"[*] Sending '{audio_path}' to server for transcription...")
    print("[-] This may take a while depending on the length of the audio. Please wait...")
    
    with open(audio_path, "rb") as f:
        # We don't set a timeout here because a 2-hour audio can take several minutes to transcribe
        resp = requests.post(server_url, files={"files": f}, timeout=None)
    
    resp.raise_for_status()
    results = resp.json()
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[*] Transcription saved to '{output_path}'.")

def main():
    parser = argparse.ArgumentParser(description="Extract audio from a video, start the transcription server, and get the text.")
    parser.add_argument("video_path", help="Path to the input video file (e.g., input.mp4)")
    parser.add_argument("--audio-out", default="temp_audio.wav", help="Temporary audio extraction path")
    parser.add_argument("--text-out", default="transcription.json", help="Output JSON path for the results")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the transcription server on")
    parser.add_argument("--keep-audio", action="store_true", help="Do not delete the temporary audio file after processing")
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        print(f"[!] Error: Video file '{args.video_path}' not found.")
        sys.exit(1)

    # 1. Extract audio
    extract_audio(args.video_path, args.audio_out)

    # 2. Start server
    print("[*] Starting the transcription server...")
    server_env = os.environ.copy()
    server_env["MAX_MODEL_LEN"] = "32768"
    server_env["MAX_NEW_TOKENS"] = "8192"
    server_env["VLLM_LIMIT_MM_PER_PROMPT"] = "audio=32768"
    
    server_cmd = [
        "uvicorn", "server:app", "--host", "0.0.0.0", "--port", str(args.port)
    ]
    # We use Popen to run the server in the background
    server_process = subprocess.Popen(server_cmd, env=server_env)

    try:
        # 3. Wait for server readiness
        wait_for_server(f"http://localhost:{args.port}/health")

        # 4. Transcribe
        transcribe_audio(args.audio_out, args.text_out, f"http://localhost:{args.port}/transcribe")
        
    except requests.exceptions.ReadTimeout:
         print("[!] Request to server timed out. Server might be overloaded.")
    except Exception as e:
         print(f"[!] Error during transcription: {e}")
    finally:
        # Stop the server after completion
        print("[*] Shutting down the server...")
        server_process.terminate()
        server_process.wait()
        
        # Cleanup temp audio unless requested otherwise
        if not args.keep_audio and os.path.exists(args.audio_out):
            os.remove(args.audio_out)
            print("[*] Cleaned up temporary audio file.")

if __name__ == "__main__":
    main()
