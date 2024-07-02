# Below is from https://www.youtube.com/watch?v=0k8wUfU7n4Q with support from https://gist.github.com/thomwolf/e9c3f978d0f82600a7c24cb0bf80d606
# Local audio recording via pyaudio -> whisper for transcriptions -> LM studio local server for LLM text -> OpenVoice TTS using audio snippet as reference

# Alternative, hacky workflow: LocalVocal in OBS doing transcription (via whisper c++) -> write to file -> Read into python -> OpenAI -> TTS

# Tech stack
# Speech to Text: Whisper (LOCAL)
# LLM: LM studio (LOCAL), OpenAI (API)
# TTS: OpenVoice (LOCAL, slow), Applio (API)

import time
timings = []

import argparse
import os
import torch
import pyaudio
import wave
import whisper
from openai import OpenAI
from api import BaseSpeakerTTS, ToneColorConverter
import se_extractor

PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'


# Function to open a file and return its contents as a string
def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

# Initialize the OpenAI client with the API key
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

# Define the name of the log file
chat_log_filename = "chatbot_conversation_log.txt"

# Function to play audio using PyAudio
def play_audio(file_path):
    # Open the audio file
    wf = wave.open(file_path, 'rb')
    # Create a PyAudio instance
    p = pyaudio.PyAudio()
    # Open a stream to play audio
    stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                    channels=wf.getnchannels(),
                    rate=wf.getframerate(),
                    output=True)
    # Read and play audio data
    data = wf.readframes(1024)
    while data:
        stream.write(data)
        data = wf.readframes(1024)
    # Stop and close the stream and PyAudio instance
    stream.stop_stream()
    stream.close()
    p.terminate()

# Command Line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--share", action='store_true', default=False, help="make link public")
args = parser.parse_args()

# Model and device setup
en_ckpt_base = 'checkpoints/base_speakers/EN'
ckpt_converter = 'checkpoints/converter'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

# Load models
start = time.perf_counter(); 
en_base_speaker_tts = BaseSpeakerTTS(f'{en_ckpt_base}/config.json', device=device)
en_base_speaker_tts.load_ckpt(f'{en_ckpt_base}/checkpoint.pth')
tone_color_converter = ToneColorConverter(f'{ckpt_converter}/config.json', device=device)
tone_color_converter.load_ckpt(f'{ckpt_converter}/checkpoint.pth')
timings.append(("BaseSpeakerTTS, ToneColorConverter", time.perf_counter() - start))

# Load speaker embeddings for English
start = time.perf_counter(); 
en_source_default_se = torch.load(f'{en_ckpt_base}/en_default_se.pth').to(device)
en_source_style_se = torch.load(f'{en_ckpt_base}/en_style_se.pth').to(device)
timings.append(("torch.load", time.perf_counter() - start))

# Main processing function
def process_and_play(prompt, style, audio_file_pth):
    global timings

    tts_model = en_base_speaker_tts
    source_se = en_source_default_se if style == 'default' else en_source_style_se
    speaker_wav = audio_file_pth
    # Process text and generate audio
    try:
        
        start = time.perf_counter()
        target_se, audio_name = se_extractor.get_se(speaker_wav, tone_color_converter, target_dir='processed', vad=True)
        timings.append(("process_and_play::se_extractor.get_se", time.perf_counter() - start))

        src_path = f'{output_dir}/tmp.wav'
        
        start = time.perf_counter()
        tts_model.tts(prompt, src_path, style, language='English')
        timings.append(("process_and_play::tts_model.tts", time.perf_counter() - start))
        
        save_path = f'{output_dir}/output.wav'
        # Run the tone color converter
        encode_message = "@MyShell"
        
        start = time.perf_counter()
        tone_color_converter.convert(audio_src_path=src_path, src_se=source_se, tgt_se=target_se, output_path=save_path, message=encode_message)
        timings.append(("process_and_play::tone_color_converter.convert", time.perf_counter() - start))

        print("Audio generated successfully.")
        play_audio(save_path)
    except Exception as e:
        print(f"Error during audio generation: {e}")

# Function to send a query to OpenAI's GPT-3.5-Turbo model, stream the response, and print each full line in yellow color. Logs the conversation to a file.
def chatgpt_streamed(user_input, system_message, conversation_history, bot_name):
    messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input}]
    temperature = 1
    streamed_completion = client.chat.completions.create(
        model="local-model",
        messages=messages,
        stream=True
    )
    full_response = ""
    line_buffer = ""
    with open(chat_log_filename, "a") as log_file:  # Open the log file in append mode
        for chunk in streamed_completion:
            delta_content = chunk.choices[0].delta.content
            if delta_content is not None:
                line_buffer += delta_content
                if '\n' in line_buffer:
                    lines = line_buffer.split('\n')
                    for line in lines[:-1]:
                        print(NEON_GREEN + line + RESET_COLOR)
                        full_response += line + '\n'
                        log_file.write(f"{bot_name}: {line}\n")  # Log the line with the bot's name
                    line_buffer = lines[-1]
        if line_buffer:
            print(NEON_GREEN + line_buffer + RESET_COLOR)
            full_response += line_buffer
            log_file.write(f"{bot_name}: {line_buffer}\n")  # Log the remaining line
    return full_response

def transcribe_with_whisper(audio_file_path):
    # Load the model
    model = whisper.load_model("base.en")  # You can choose different model sizes like 'tiny', 'base', 'small', 'medium', 'large'
    # Transcribe the audio
    result = model.transcribe(audio_file_path)
    return result["text"]

def list_audio_devices():
    p = pyaudio.PyAudio()
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        print(f"Device {i}: {info['name']}")
    p.terminate()

# Function to record audio from the microphone and save to a file
def record_audio(file_path, device_index=None):
    global timings

    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024, input_device_index=device_index)
    frames = []
    print("Recording...")
    try:
        while True:
            data = stream.read(1024)
            frames.append(data)
    except KeyboardInterrupt:
        pass
    print("Recording stopped.")

    start = time.perf_counter()
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(file_path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    wf.setframerate(16000)
    wf.writeframes(b''.join(frames))
    wf.close()
    timings.append(("record_audio::clean up", time.perf_counter() - start))

# New function to handle a conversation with a user
def user_chatbot_conversation():
    global timings

    conversation_history = []
    system_message = open_file("chatbot1.txt")
    while True:
        audio_file = "temp_recording.wav"
        record_audio(audio_file, device_index)

        start = time.perf_counter()
        user_input = transcribe_with_whisper(audio_file)
        timings.append(("user_chatbot_conversation::transcribe_with_whisper", time.perf_counter() - start))

        os.remove(audio_file)  # Clean up the temporary audio file

        print(user_input.lower())
        if "stop conversation" in user_input.lower():  # Say 'exit' to end the conversation
            for msg, val in timings:
                print(f'{msg}: {val}')
            break

        print(CYAN + "You:", user_input + RESET_COLOR)
        conversation_history.append({"role": "user", "content": user_input})
        print(PINK + "Chatbot:" + RESET_COLOR)

        start = time.perf_counter()
        chatbot_response = chatgpt_streamed(user_input, system_message, conversation_history, "Chatbot")
        timings.append(("user_chatbot_conversation::chatgpt_streamed", time.perf_counter() - start))

        conversation_history.append({"role": "assistant", "content": chatbot_response})
        prompt2 = chatbot_response
        style = "friendly" # ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']
        audio_file_pth2 = "ref.mp3"
        process_and_play(prompt2, style, audio_file_pth2)
        if len(conversation_history) > 20:
            conversation_history = conversation_history[-20:]



start = time.perf_counter()
list_audio_devices()
timings.append(("user_chatbot_conversation::chatgpt_streamed", time.perf_counter() - start))

device_index = int(input("Please enter the device index you want to use: "))

user_chatbot_conversation()
