"""
SAFE-T (speech analysis for emergency response technology) has 131 hrs
(labelled and unlabelled) of single-channel 48 kHz training data.
Most of the speakers are native English speakers. The participants
are playing the game of Flashpoint fire rescue. The recordings do not
have overlap, little reverberation but have significant noise. The noise
is artificial and the SNR varies with time. The noise level varies
from 0-14db or 70-85 dB. The noises are car ambulances, rain, or similar sounds.
There are a total of 87 speakers.
We currently support the following LDC packages:

SAFE-T
  Speech       LDC2020E10
  Transcripts  LDC2020E09

This data is not available for free - your institution needs to have an LDC subscription.
"""

from collections import defaultdict

import re
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Union
from cytoolz import sliding_window

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, check_and_rglob, recursion_limit

data_directories = {
        'audio': ['LDC2020E10', 'LDC2019E37' 'LDC2019E53'],
        'transcription': ['LDC2020E09', 'LDC2019E36', 'LDC2019E53']
}

PARTITIONS = {
    'train': {
        'audio': ['LDC2020E10', 'LDC2019E37'],
        'transcription': ['LDC2020E09', 'LDC2019E36']
    },
    'dev':   {
        'audio': ['LDC2019E53'],
        'transcription': ['LDC2019E53']
    }
}

def case_normalize(w):
    if w.startswith('~'):
        return w.upper()
    else:
        return w.lower()


def process_transcript(transcript):
    tmp = re.sub(r'extreme\s+background', 'extreme_background', transcript)
    tmp = re.sub(r'foreign\s+lang=', 'foreign_lang=', tmp)
    tmp = re.sub(r'\)\)([^\s])', ')) \1', tmp)
    tmp = re.sub(r'[.,!?]', ' ', tmp)
    tmp = re.sub(r' -- ', ' ', tmp)
    tmp = re.sub(r' --$', '', tmp)
    x = re.split(r'\s+', tmp)
    old_x = x
    x = list()

    w = old_x.pop(0)
    while old_x:
        if w.startswith(r'(('):
            while old_x and not w.endswith('))'):
                w2 = old_x.pop(0)
                w += ' ' + w2
            x.append(w)
            if old_x:
                w = old_x.pop(0)
        elif w.startswith(r'<'):
            #this is very simplified and assumes we will not get a starting tag
            #alone
            while old_x and not w.endswith('>'):
                w2 = old_x.pop(0)
                w += ' ' + w2
            x.append(w)
            if old_x:
                w = old_x.pop(0)
        elif w.endswith(r'))'):
            if old_x:
                w = old_x.pop(0)
        else:
            x.append(w)
            if old_x:
                w = old_x.pop(0)

    if not x:
        return None
    if len(x) == 1 and x[0] in ('<background>', '<extreme_background>'):
        return None

    out_x = list()
    for w in x:
        w = case_normalize(w)
        out_x.append(w)
    return ' '.join(out_x)


def prepare_safet(
        corpus_dir: Pathlike,
        output_dir: Optional[Pathlike] = None
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:
    """
    Prepare manifests for safet corpus.
    We create two manifests: one with recordings, one with segments supervisions.

    :param corpus_dir: Path to ``LDC2020E10`` package.
    :param output_dir: Directory where the manifests should be written. Can be omitted to avoid writing.
    :return: A dict with manifests. The keys are: ``{'recordings', 'segments'}``.
    """

    corpus_dir = Path(corpus_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    dataset_parts = ['train', 'dev']
    manifests = defaultdict(dict)
    for part in dataset_parts:
        recordings = []
        supervisions = []
        supervision_id = 0
        # create recordings list with sampling_rate, num_samples
        # duration, location and id.
        wav_dir = corpus_dir / f'{part}' / 'audio_dir'
        transcript_dir = corpus_dir / f'{part}' / 'transcript_dir'
        audio_paths = check_and_rglob(wav_dir, '*.wav')
        for audio_path in audio_paths:
          recording = Recording.from_file(audio_path)
          recordings.append(recording)

        # Create supervision list with transcription, speaker_id,
        # language_id, time and recording information.
        # get list of transcript path.
        transcript_paths = check_and_rglob(transcript_dir, '*.tsv')
        for transcript_path in transcript_paths:
            # get basename of the file and add '_mixed' suffix to it
            # (this is also the recording_id)
            if part == 'dev':
                recording_id = Path(transcript_path).stem + '_dev_mixed'
            else:
                recording_id = Path(transcript_path).stem + '_mixed'
            with open(transcript_path, 'r', encoding='utf-8') as f:
                for line in f.readlines():
                    line_parts = line.split()
                    start_time = float(line_parts[0])
                    end_time = float(line_parts[1])
                    # subtract 0.1 sec to avoid supervision to be larger
                    # than recording
                    duration = end_time - start_time - 0.1
                    speaker_id = line_parts[2][:-1]
                    transcription = " ".join(line_parts[3:])
                    #cleaned_transcrition = transcription
                    cleaned_transcrition = process_transcript(transcription)
                    # do not use utterances which have empty cleaned transcript
                    if not cleaned_transcrition:
                        continue
                    # do not use utterances which do not have speaker label
                    if 'background' in speaker_id:
                        continue
                    supervision_id += 1
                    segment = SupervisionSegment(
                        id=supervision_id,
                        recording_id=recording_id,
                        start=float(start_time),
                        duration=duration,
                        channel=0,
                        language='English',
                        speaker=speaker_id,
                        text=cleaned_transcrition.strip()
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
