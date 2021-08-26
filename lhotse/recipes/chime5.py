"""
CHiME-5
  http://spandh.dcs.shef.ac.uk/chime_challenge/CHiME5/data.html
  CHiME5/audio/{dev,eval,train}
  CHiME5/transcriptions/{dev,eval,train}
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


def hms_to_seconds(hms):
    hour = hms.split(':')[0]
    minute = hms.split(':')[1]
    second = hms.split(':')[2]

    # total seconds
    seconds = float(hour) * 3600 + float(minute) * 60 + float(second)
    
    return seconds


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

    dataset_parts = ['dev', 'train']
    manifests = defaultdict(dict)
    for part in dataset_parts:
        recordings = []
        supervisions = []
        wav_dir = corpus_dir / 'audio' / f'{part}' 
        transcript_dir = corpus_dir / 'transcriptions' / f'{part}' 
        audio_paths = check_and_rglob(wav_dir, '*.wav')
        #? binaural microphone will have two channels
        for audio_path in audio_paths:
          recording = Recording.from_file(audio_path)
          recordings.append(recording)

        transcript_paths = check_and_rglob(transcript_dir, '*.json')
        for transcript_path in transcript_paths:
            with open(transcript_path, 'r', encoding="utf-8") as f:
                j = json.load(f)
                for x in j:
                    if '[redacted]' not in x['words']:
                        session_id = x['session_id']
                        speaker_id = x['speaker']
                        mictype = 'original'
                        
                        start_time = x['start_time'][mictype]
                        end_time = x['end_time'][mictype]
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
                                        .replace('!', '').lower()
                        #? remove multiple spaces
                        transcription = " ".join(transcription.split())
                        uttid = speaker_id + '_' + session_id
                        recording_id = session_id + '_' + speaker_id
                        #? print(recording_id)
                        #? print(duration)
                        #? print(uttid)
                        #? In several utterances, there are inconsistency in the time stamp (the end time is earlier than the start time) We just ignored such utterances.
                        if end_time > start_time:
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


def main():
    prepare_chime('/export/common/data/corpora/CHiME5/',
    '/exp/aarora/icefall_work_env/lhotse_output/chime')


if __name__ == '__main__':
    main()