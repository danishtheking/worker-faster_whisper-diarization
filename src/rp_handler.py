"""
rp_handler.py for runpod worker

rp_debugger:
- Utility that provides additional debugging information.
The handler must be called with --rp_debugger flag to enable it.
"""
import base64
import subprocess
import tempfile
from pathlib import Path

from pyannote.audio import Pipeline
from rp_schema import INPUT_VALIDATIONS
from runpod.serverless.utils import download_files_from_urls, rp_cleanup, rp_debugger
from runpod.serverless.utils.rp_validator import validate
import runpod
import predict
import torch
import numpy as np


np.NAN = np.nan

MODEL = predict.Predictor()
MODEL.setup()


def base64_to_tempfile(base64_file: str) -> str:
    '''
    Convert base64 file to tempfile.

    Parameters:
    base64_file (str): Base64 file

    Returns:
    str: Path to tempfile
    '''
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_file.write(base64.b64decode(base64_file))

    return temp_file.name


def _to_wav(fpath):
    path = Path(fpath)
    new_name = path.name.split('.')[0] + '.wav'
    new_path = path.parent / new_name
    subprocess.run([
        'ffmpeg',
        '-i', str(fpath),
        '-ar', '16000',
        '-ac', '1',
        '-c:a', 'pcm_s16le',
        str(new_path)
    ])
    return new_path


def diarize(fpath):
    if not str(fpath).lower().endswith('.wav'):
        fpath = _to_wav(fpath)

    resp = {'segments': []}
    # PyTorch 2.6 changed weights_only default to True, which breaks pyannote
    # Force weights_only=False for trusted local model files
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False  # force override regardless of existing value
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load
    try:
        pipeline = Pipeline.from_pretrained('config.yaml')
    finally:
        torch.load = _original_torch_load
    pipeline.to(torch.device('cuda'))
    dia = pipeline(fpath)

    speakers = {}
    for turn, _, speaker in dia.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = len(speakers)  # assign ordered index

        segdata = {'start': turn.start, 'end': turn.end, 'speaker': speakers[speaker]}
        resp['segments'].append(segdata)

    return resp


@rp_debugger.FunctionTimer
def run_whisper_job(job):
    '''
    Run inference on the model.

    Parameters:
    job (dict): Input job containing the model parameters

    Returns:
    dict: The result of the prediction
    '''
    job_input = job['input']

    with rp_debugger.LineTimer('validation_step'):
        input_validation = validate(job_input, INPUT_VALIDATIONS)

        if 'errors' in input_validation:
            return {"error": input_validation['errors']}
        job_input = input_validation['validated_input']

    if not job_input.get('audio', False) and not job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64'}

    if job_input.get('audio', False) and job_input.get('audio_base64', False):
        return {'error': 'Must provide either audio or audio_base64, not both'}

    if job_input.get('audio', False):
        with rp_debugger.LineTimer('download_step'):
            audio_input = download_files_from_urls(job['id'], [job_input['audio']])[0]

    if job_input.get('audio_base64', False):
        audio_input = base64_to_tempfile(job_input['audio_base64'])

    with rp_debugger.LineTimer('prediction_step'):
        resp = MODEL.predict(
            audio=audio_input,
            model_name=job_input["model"],
            transcription=job_input["transcription"],
            translation=job_input["translation"],
            translate=job_input["translate"],
            language=job_input["language"],
            temperature=job_input["temperature"],
            best_of=job_input["best_of"],
            beam_size=job_input["beam_size"],
            patience=job_input["patience"],
            length_penalty=job_input["length_penalty"],
            suppress_tokens=job_input.get("suppress_tokens", "-1"),
            initial_prompt=job_input["initial_prompt"],
            condition_on_previous_text=job_input["condition_on_previous_text"],
            temperature_increment_on_fallback=job_input["temperature_increment_on_fallback"],
            compression_ratio_threshold=job_input["compression_ratio_threshold"],
            logprob_threshold=job_input["logprob_threshold"],
            no_speech_threshold=job_input["no_speech_threshold"],
            enable_vad=job_input["enable_vad"],
            word_timestamps=job_input["word_timestamps"],
            repetition_penalty=job_input["repetition_penalty"],
            no_repeat_ngram_size=job_input["no_repeat_ngram_size"],
        )

    if job_input['diarize']:
        resp['diarization'] = diarize(audio_input)

    with rp_debugger.LineTimer('cleanup_step'):
        rp_cleanup.clean(['input_objects'])

    return resp


runpod.serverless.start({"handler": run_whisper_job})
