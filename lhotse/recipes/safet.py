"""
SAFE-T
  Speech       LDC2020E10
  Transcripts  LDC2020E09
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


def prepare_safet(
        corpus_dir,
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
                    # do not use utterances which do not have speaker label
                    if speaker_id.startswith('background'):
                        continue
                    supervision_id = supervision_id + 1
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
                        text=transcription
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
    