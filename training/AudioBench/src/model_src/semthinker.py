import os
import re

# add parent directory to sys.path
import sys
sys.path.append('.')
sys.path.append('../')
import logging
import numpy as np
import torch

from tqdm import tqdm

import soundfile as sf

from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor

import tempfile


# =  =  =  =  =  =  =  =  =  =  =  Logging Setup  =  =  =  =  =  =  =  =  =  =  =  =  =
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
# =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =

def semthinker_model_loader(self, model_name):
    if model_name == "semthinker":
        model_path = "gijs/semthinker"
    elif model_name == "semthinker-sem":
        model_path = "gijs/semthinker-sem"
    elif "/checkpoints/" in model_name:
        model_path = model_name
    else:
        raise ValueError(f"Model name {model_name} not found")

    self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct")
    self.model = Qwen2AudioForConditionalGeneration.from_pretrained(model_path, device_map="auto")
    logger.info("Model loaded: {}".format(model_path))


def post_process_qwen2_asr(model_output):
    
    match = re.search(r'"((?:\\.|[^"\\])*)"', model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    if ":'" in model_output:
        model_output = "'" + model_output.split(":'")[1]
    elif ": '" in model_output:
        model_output = "'" + model_output.split(": '")[1]

    # Find the longest match of ''
    match = re.search(r"'(.*)'", model_output)
    if match:
        model_output = match.group(1)
    else:
        model_output = model_output

    return model_output


def semthinker_model_generation(self, input):

    audio_array    = input["audio"]["array"]
    sampling_rate  = input["audio"]["sampling_rate"]
    audio_duration = len(audio_array) / sampling_rate

    os.makedirs('tmp', exist_ok=True)
    
    assert input["task_type"] != "ASR", "ASR task is not supported for semthinker model."

    if audio_duration > 30:
        logger.info('Audio duration is more than 30 seconds. Taking first 30 seconds.')
        #audio_path = f"tmp/audio_{self.dataset_name}.wav"
        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array[:30 * sampling_rate], sampling_rate)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path.name},
                {"type": "text", "text": "Describe the audio in detail."},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                # BytesIO(urlopen(ele['audio_url']).read()), 
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        try:
            output = response.split("<answer>")[1].replace("</answer>", "").strip()
        except:
            print("Output: ", response)
            output = response


    else: 
        if audio_duration < 1:
            logger.info('Audio duration is less than 1 second. Padding the audio to 1 second.')
            audio_array = np.pad(audio_array, (0, sampling_rate), 'constant')

        audio_path = tempfile.NamedTemporaryFile(suffix=".wav", prefix="audio_", delete=False)
        sf.write(audio_path.name, audio_array, sampling_rate)

        conversation = [
            {"role": "user", "content": [
                {"type": "audio", "audio_url": audio_path.name},
                {"type": "text", "text": "Describe the audio in detail."},
            ]},
        ]

        text = self.processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios = []
        for message in conversation:
            if isinstance(message["content"], list):
                for ele in message["content"]:
                    if ele["type"] == "audio":
                        audios.append(
                            librosa.load(
                                ele['audio_url'],
                                sr=self.processor.feature_extractor.sampling_rate)[0]
                        )

        inputs = self.processor(text=text, audios=audios, sampling_rate=self.processor.feature_extractor.sampling_rate, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")

        generate_ids = self.model.generate(**inputs, max_length=1024)
        generate_ids = generate_ids[:, inputs.input_ids.size(1):]

        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        try:
            output = response.split("<answer>")[1].replace("</answer>", "").strip()
        except:
            print("Output: ", response)
            output = response

    return output

