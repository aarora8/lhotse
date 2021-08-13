"""
SAFE-T (speech analysis for emergency response technology) has 131 hrs
(labelled and unlabelled) of single-channel 48 kHz training data.
Most of the speakers are native English speakers. The participants
are playing the game of Flashpoint fire rescue. The recordings do not
have overlap, little reverberation but have significant noise. The noise
is artificial and the SNR varies with time. The noise level varies
from 0-14db or 70-85 dB. The noises are car ambulances, rain, or similar sounds.
There are a total of 87 speakers.

SAFE-T
  Speech       LDC2020E10
  Transcripts  LDC2020E09
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


WORDLIST = dict()
UNK = '<UNK>'
REPLACE_UNKS = True

def case_normalize(w):
    # this is for POI
    # but we should add it into the lexicon
    if w.startswith('~'):
        return w.upper()
    else:
        return w.upper()

def process_transcript(transcript):
    global WORDLIST
    if_only_unk = True
    # https://www.programiz.com/python-programming/regex
    # [] for set of characters you with to match
    # eg. [abc] --> will search for a or b or c
    # "." matches any single character
    # "$" to check if string ends with a certain character 
    # eg. "a$" should end with "a"
    # replace <extreme background> with <extreme_background>
    # replace <foreign lang="Spanish">fuego</foreign> with foreign_lang=
    # remove "[.,!?]"
    # remove " -- "
    # remove " --" --> strings that ends with "-" and starts with " "
    # \s+ markers are – that means “any white space character, one or more times”
    tmp = re.sub(r'<extreme background>', '', transcript)
    tmp = re.sub(r'<background>', '', transcript)
    tmp = re.sub(r'foreign\s+lang=', 'foreign_lang=', tmp)
    tmp = re.sub(r'\(\(', '', tmp)
    tmp = re.sub(r'\)\)', '', tmp)
    tmp = re.sub(r'[.,!?]', ' ', tmp)
    tmp = re.sub(r' -- ', ' ', tmp)
    tmp = re.sub(r' --$', '', tmp)
    list_words = re.split(r'\s+', tmp)

    out_list_words = list()
    for w in list_words:
        w = w.strip()
        w = case_normalize(w)
        if w == "":
            continue
        elif w in WORDLIST:
            out_list_words.append(w)
            if_only_unk = False
        else:
            out_list_words.append(UNK)

    if if_only_unk:
        out_list_words = ''
    return ' '.join(out_list_words)


def read_lexicon_words(lexicon):
    with open(lexicon, 'r', encoding='utf-8') as f:
        for line in f:
            line = re.sub(r'(?s)\s.*', '', line)
            WORDLIST[line] = 1


def prepare_safet(
        corpus_dir: Pathlike,
        lexicon_dir:Pathlike,
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
    lexicon_dir = Path(lexicon_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    lexicon = lexicon_dir / 'lexicon_raw_nosil.txt'
    read_lexicon_words(lexicon)
    dataset_parts = ['dev', 'dev_clean', 'train']
    manifests = defaultdict(dict)
    for part in dataset_parts:
        recordings = []
        supervisions = []
        supervision_id = 0
        # create recordings list with sampling_rate, num_samples
        # duration, location and id.
        if 'dev' in part:
            wav_dir = corpus_dir / 'dev' / 'audio_dir'
            transcript_dir = corpus_dir / 'dev' / 'transcript_dir'
        else:
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
            if 'dev' in part:
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
                    if part == 'dev':
                        cleaned_transcrition = transcription
                    else:
                        cleaned_transcrition = process_transcript(transcription)
                    # do not use utterances which have empty cleaned transcript
                    if not cleaned_transcrition:
                        continue
                    # do not use utterances which do not have speaker label
                    if speaker_id.startswith('background'):
                        continue
                    supervision_id += 1
                    speaker_id = str(speaker_id).zfill(6)
                    supervision_id_str = str(supervision_id).zfill(6)
                    uttid =f'{speaker_id}_{supervision_id_str}'
                    segment = SupervisionSegment(
                        id=uttid,
                        recording_id=recording_id,
                        start=float(start_time),
                        duration=duration,
                        channel=0,
                        language='English',
                        speaker=speaker_id,
                        text=cleaned_transcrition
                    )
                    supervisions.append(segment)
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        validate_recordings_and_supervisions(recording_set, supervision_set)
        if output_dir is not None:
            supervision_set.to_json(output_dir / f'supervisions_safet_{part}.json')
            recording_set.to_json(output_dir / f'recordings_safet_{part}.json')
        manifests[part] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }
    return manifests
