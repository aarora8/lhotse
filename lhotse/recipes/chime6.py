"""
CHiME-6
  http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/data.html
  CHiME6/audio/{dev,eval,train}
  CHiME6/transcriptions/{dev,eval,train}
  end_time, start_time, words, speaker, session_id, reference and location
"""

from collections import defaultdict
import logging
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from cytoolz import sliding_window

from lhotse.utils import Pathlike, check_and_rglob, recursion_limit
from lhotse import compute_num_samples, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available
import json
import argparse
import logging
import sys
import copy
from datetime import timedelta
from decimal import Decimal

SAMPLE_RATE = 16000

Session2Microphones = {
    'S01': ['U01','U02','U04','U05','U06'],
    'S05': ['U01','U02','U05','U06'],
    'S09': ['U01','U02','U03','U04','U06'],
    'S22': ['U01','U02','U04','U05','U06']
}


def prepare_chime(
        corpus_dir,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for CHiME corpus.
    We create two manifests: one with recordings, one with segments supervisions.
    :param corpus_dir: Path to chime '/export/common/data/corpora/CHiME5'.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :return: A dict with manifests. keys: ``{'recordings', 'segments'}``.
    """

    corpus_dir = Path(corpus_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    #?dataset_parts = ['dev', 'eval', 'train']
    dataset_parts = ['dev', 'train']
    manifests = defaultdict(dict)
    for part in dataset_parts:
        recordings = []
        supervisions = []
        supervision_id = 0
        microphonetype_list = get_microphonetype_list(part)
        wav_dir = corpus_dir / 'audio' / f'{part}' 
        transcript_dir = corpus_dir / 'transcriptions' / f'{part}' 
        audio_paths = check_and_rglob(wav_dir, '*.wav')
        #? binaural microphone will have two channels
        for audio_path in audio_paths:
          recording = Recording.from_file(audio_path)
          recordings.append(recording)
        #  recording1 = copy.deepcopy(recording)
        #   if len(recording.sources[0].channels) == 2:
        #       recording.id = recording.id + '.L'
        #       recording.sources[0].channels = [0]
        #       recordings.append(recording)

        transcript_paths = check_and_rglob(transcript_dir, '*.json')
        for transcript_path in transcript_paths:
            with open(transcript_path, 'r', encoding="utf-8") as f:
                j = json.load(f)
                for x in j:
                    for microphonetype in microphonetype_list:
                        if '[redacted]' not in x['words']:
                            session_id = x['session_id']
                            speaker_id = x['speaker']
                            if microphonetype == 'worn':
                                mictype = 'original'
                                recording_id_list = get_recording_id_list(mictype, session_id, speaker_id)
                            elif microphonetype == 'ref':
                                mictype = x['ref']
                                recording_id_list = get_recording_id_list(mictype, session_id, speaker_id)
                            else:
                                mictype = microphonetype
                                recording_id_list = get_recording_id_list(mictype, session_id, speaker_id)

                            for recording_id in recording_id_list:
                                
                                supervision_id = supervision_id + 1
                                end_time, start_time, uttid, duration, transcription = get_supervision_details(x, supervision_id, speaker_id, session_id)
                                #? In several utterances, there are inconsistency in the time stamp (the end time is earlier than the start time) We just ignored such utterances.
                                if end_time == None or start_time == None or uttid == None or duration == None or transcription == None:
                                    continue

                                if end_time <= start_time:
                                    continue
                                segment = SupervisionSegment(
                                    id=uttid,
                                    recording_id=recording_id,
                                    start=start_time,
                                    duration=duration,
                                    channel=0,
                                    language='English',
                                    speaker=speaker_id,
                                    text=transcription
                                )
                                supervisions.append(segment)

        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)
        if output_dir is not None:
            supervision_set.to_json(output_dir / f'supervisions_{part}.json')
            recording_set.to_json(output_dir / f'recordings_{part}.json')
        manifests[part] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }
    return manifests


def get_recording_id_list(mictype, session_id, speaker_id):
    recording_id_list = []
    channel_list = ['CH1', 'CH2', 'CH3', 'CH4']
    # we are not using S12 duration mismatch for some utterances
    if session_id == 'S12':
        return recording_id_list

    if mictype == 'original':
        # recording_id = session_id + '_' + speaker_id + '.L'
        # recording_id_list.append(recording_id)
        # recording_id = session_id + '_' + speaker_id + '.R'
        # recording_id_list.append(recording_id)
        recording_id = session_id + '_' + speaker_id
        recording_id_list.append(recording_id)
        return recording_id_list
    else:
        if session_id in Session2Microphones:
            if mictype in Session2Microphones[session_id]:
                for channel in channel_list:
                    recording_id = session_id + '_' + mictype + '.' + channel
                    recording_id_list.append(recording_id)
        else:
            for channel in channel_list:
                    recording_id = session_id + '_' + mictype + '.' + channel
                    recording_id_list.append(recording_id)
        return recording_id_list


# def get_microphonetype_list(dataset_part):
#     if dataset_part == 'dev':
#         microphonetype_list = ['worn', 'ref', 'U01', 'U02', 'U03', 'U04', 'U05', 'U06']
#     elif dataset_part == 'eval':
#         microphonetype_list = ['U01', 'U02', 'U03', 'U04', 'U05', 'U06']
#     else:
#         microphonetype_list = ['worn', 'U01', 'U02', 'U03', 'U04', 'U05', 'U06']
    
#     return microphonetype_list

def get_microphonetype_list(dataset_part):
    if dataset_part == 'dev':
        microphonetype_list = ['worn']
    else:
        microphonetype_list = ['worn']
    
    return microphonetype_list

def hms_to_seconds(hms):
    hour = hms.split(':')[0]
    minute = hms.split(':')[1]
    second = hms.split(':')[2]

    # total seconds
    seconds = float(hour) * 3600 + float(minute) * 60 + float(second)
    
    return seconds


def to_samples(time: str):
    "mapping time in string to int, as mapped in pb_chime5"
    "see https://github.com/fgnt/pb_chime5/blob/master/pb_chime5/database/chime5/get_speaker_activity.py"
    hours, minutes, seconds = [t for t in time.split(':')]
    hours = int(hours)
    minutes = int(minutes)
    seconds = Decimal(seconds)

    seconds_samples = seconds * SAMPLE_RATE
    samples = (
        hours * 3600 * SAMPLE_RATE
        + minutes * 60 * SAMPLE_RATE
        + seconds_samples
    )
    return int(samples)


def get_supervision_details(x, supervision_id, speaker_id, session_id):
    try:
        start_time = x['start_time']
        end_time = x['end_time']
        #? convert to seconds, e.g., 1:10:05.55 -> 3600 + 600 + 5.55 = 4205.55
        start_time = hms_to_seconds(start_time)
        end_time = hms_to_seconds(end_time)
        duration = end_time - start_time

        #? remove meta chars and convert to lower
        transcription = x['words'].replace('"', '')\
                .replace('.', '')\
                .replace('?', '')\
                .replace(',', '')\
                .replace(':', '')\
                .replace(';', '')\
                .replace('!', '')
        #? remove multiple spaces
        transcription = " ".join(transcription.split())
        supervision_id_str = str(supervision_id).zfill(6)
        uttid =f'{speaker_id}_{session_id}_{supervision_id_str}'
    except Exception:
        return None, None, None, None, None

    return end_time, start_time, uttid, duration, transcription


# def main():
#     prepare_chime('/exp/aarora/CHiME6_data_prep/CHiME6/',
#     '/exp/aarora/icefall_work_env/icefall/egs/chime6/ASR/data/manifests')


# if __name__ == '__main__':
#     main()
