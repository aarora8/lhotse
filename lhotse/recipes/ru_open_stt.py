from collections import defaultdict
import logging
from itertools import chain, repeat
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from cytoolz import sliding_window
import os
from tqdm.auto import tqdm

from lhotse.utils import Pathlike, check_and_rglob, recursion_limit
from lhotse import compute_num_samples, fix_manifests, validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.recipes.utils import manifests_exist, read_manifests_if_cached
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, is_module_available


def prepare_ru_open_stt(
        corpus_dir, output_dir
) -> Dict[str, Union[RecordingSet, SupervisionSet]]:

    corpus_dir = Path(corpus_dir)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    manifests = defaultdict(dict)
    recordings = []
    supervisions = []
    supervision_id = 0
    #! create recordings list with sampling_rate, duration
    dataset_parts = ['val', 'train']
    for part in dataset_parts:
        wav_dir = corpus_dir / f'{part}'
        transcript_dir = corpus_dir / f'{part}'
        if part == 'val':
            audio_paths = check_and_rglob(wav_dir, '*.wav')
        else:
            audio_paths = check_and_rglob(wav_dir, '*.opus')
        
        for audio_path in audio_paths:
            recording = Recording.from_file(audio_path)
            recordings.append(recording)

        #! Create supervision list with transcription,
        transcript_paths = check_and_rglob(transcript_dir, '*.txt')
        for transcript_path in transcript_paths:
            audio_path = "/".join(str(transcript_path).strip().split('/')[0:-1])
            recording_id = Path(transcript_path).stem
            if part == 'val':
                recording_id = recording_id + '.wav'
            else:
                recording_id = recording_id + '.opus'

            audio_file = os.path.join(audio_path, recording_id)
            #recording = Recording.from_file(audio_file)
            transcript_text = transcript_path.read_text().splitlines()[0]
            supervision_id = supervision_id + 1
            segment = SupervisionSegment(
                id=Path(transcript_path).stem,
                recording_id=Path(transcript_path).stem,
                start=0.0,
                duration=5,
                channel=0,
                language='Russian',
                text=transcript_text
            )
            supervisions.append(segment)
        recording_set = RecordingSet.from_recordings(recordings)
        supervision_set = SupervisionSet.from_segments(supervisions)
        #validate_recordings_and_supervisions(recording_set, supervision_set)
        if output_dir is not None:
            supervision_set.to_json(output_dir / f'supervisions_{part}.json')
            recording_set.to_json(output_dir / f'recordings_{part}.json')
        manifests[part] = {
                'recordings': recording_set,
                'supervisions': supervision_set
            }
    return manifests


def main():
    prepare_ru_open_stt('/exp/aarora/storage/ru_open_stt/unzip/',
    '/exp/aarora/icefall_work_env/lhotse_output/ru_open_stt')


if __name__ == '__main__':
    main()
