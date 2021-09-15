import itertools
import logging
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import html
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Union
import gzip
import shutil
import re
from tqdm.auto import tqdm

from lhotse import validate_recordings_and_supervisions
from lhotse.audio import AudioSource, Recording, RecordingSet
from lhotse.supervision import SupervisionSegment, SupervisionSet
from lhotse.utils import Pathlike, Seconds, urlretrieve_progress

PARTITIONS = {
    'full-corpus-asr': {
        'train': [
            'EN2001a','EN2001b','EN2001d','EN2001e','EN2003a','EN2004a','EN2005a','EN2006a','EN2006b','EN2009b','EN2009c','EN2009d','ES2002a','ES2002b','ES2002c','ES2002d','ES2003a','ES2003b','ES2003c','ES2003d','ES2005a','ES2005b','ES2005c','ES2005d','ES2006a','ES2006b','ES2006c','ES2006d','ES2007a','ES2007b','ES2007c','ES2007d','ES2008a','ES2008b','ES2008c','ES2008d','ES2009a','ES2009b','ES2009c','ES2009d','ES2010a','ES2010b','ES2010c','ES2010d','ES2012a','ES2012b','ES2012c','ES2012d','ES2013a','ES2013b','ES2013c','ES2013d','ES2014a','ES2014b','ES2014c','ES2014d','ES2015a','ES2015b','ES2015c','ES2015d','ES2016a','ES2016b','ES2016c','ES2016d','IB4005','IN1001','IN1002','IN1005','IN1007','IN1008','IN1009','IN1012''IN1013','IN1014','IN1016','IS1000a','IS1000b','IS1000c','IS1000d','IS1001a','IS1001b','IS1001c','IS1001d','IS1002b','IS1002c','IS1002d','IS1003a','IS1003b','IS1003c','IS1003d','IS1004a','IS1004b','IS1004c','IS1004d','IS1005a','IS1005b','IS1005c','IS1006a','IS1006b','IS1006c','IS1006d','IS1007a','IS1007b','IS1007c','IS1007d','TS3005a','TS3005b','TS3005c','TS3005d','TS3006a','TS3006b','TS3006c','TS3006d','TS3007a','TS3007b','TS3007c','TS3007d','TS3008a','TS3008b','TS3008c','TS3008d','TS3009a','TS3009b','TS3009c','TS3009d','TS3010a','TS3010b','TS3010c','TS3010d','TS3011a','TS3011b','TS3011c','TS3011d','TS3012a','TS3012b','TS3012c','TS3012d'
            ],
        'dev': [
                'ES2011a','ES2011b','ES2011c','ES2011d','IB4001','IB4002','IB4003','IB4004','IB4010','IB4011','IS1008a','IS1008b','IS1008c','IS1008d','TS3004a','TS3004b','TS3004c','TS3004d'
            ],
        'test': [
                'EN2002a','EN2002b','EN2002c','EN2002d','ES2004a','ES2004b','ES2004c','ES2004d','IS1009a','IS1009b','IS1009c','IS1009d','TS3003a','TS3003b','TS3003c','TS3003d'
            ]
    }
}
MICS = ['ihm','ihm-mix','sdm','mdm']

class AmiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds


def parse_ami_annotations() -> Dict[str, List[SupervisionSegment]]:

    "<meetingID> <channel> <spkr> <start time> <end time> <transcripts>"
    annotations_url = 'https://raw.githubusercontent.com/aarora8/lhotse_input/main/transcripts2'
    annotations_path = '/exp/aarora/icefall_work_env/other/lhotse_output/ami/transcript2'
    urllib.request.urlretrieve(annotations_url, filename=annotations_path)
    annotations_handle = open(annotations_path, 'r', encoding='utf8')
    annotations_data_vect = annotations_handle.read().strip().split("\n")
    annotations = defaultdict(dict)
    for line in annotations_data_vect:
        parts = line.strip().split()
        meeting_id = parts[0]
        spk = parts[2]
        channel = parts[1]
        st_time = float(parts[3])
        end_time = float(parts[4])
        text = " ".join(parts[5:])
        key = (meeting_id, spk)
        if key not in annotations:
            annotations[key] = []
        annotations[key].append(
            AmiSegmentAnnotation(
                text=text,
                speaker=spk,
                gender=spk[0],
                begin_time=float(st_time),
                end_time=float(end_time),
            )
        )
    return annotations


def prepare_ami(
    data_dir: Pathlike,
    output_dir: Pathlike,
    mic: Optional[str] = "ihm",
    partition: Optional[str] = "full-corpus-asr",
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param data_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic to use.
    :param partition: str {'full-corpus','full-corpus-asr','scenario-only'}, AMI official data split
    :return: a Dict whose key is ('train', 'dev', 'eval'), and the values are dicts of manifests under keys 'recordings' and 'supervisions'.
    """
    data_dir = Path(data_dir)
    assert data_dir.is_dir(), f"No such directory: {data_dir}"
    assert mic in MICS, f"Mic {mic} not supported"
    assert partition in PARTITIONS, f"Partition {partition} not supported"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Parsing AMI annotations")
    #? key = (meeting_id, spk, channel)
    #? text, speaker, gender, begin_time, end_time
    annotations = parse_ami_annotations()
    annotation_by_id = {(key[0]): annot for key, annot in annotations.items()}
    #? Audio
    logging.info("Preparing recording manifests")
    wav_dir = data_dir
    audio_paths = (wav_dir.rglob("*Array1-01.wav"))
    recordings = []
    for audio_path in audio_paths:
        recording = Recording.from_file(audio_path)
        recordings.append(recording)
    recording_set = RecordingSet.from_recordings(recordings)
    #? Supervisions
    logging.info("Preparing supervision manifests")
    segments = []
    for recording in recording_set:
        source = recording.sources[0].source
        channel = source.split('/')[-1].split('-')[-1].split('.')[0]
        meetingid = recording.id.split('.')[0]
        channel = int(channel.strip())
        meetingid = str(meetingid.strip())
        
        annotation = annotation_by_id.get(meetingid)
        if annotation is None:
            logging.warning(
            f"No annotation found for recording {meetingid} "
            f"(file {channel})")
            continue
        for seg_idx, seg_info in enumerate(annotation):
            duration = seg_info.end_time - seg_info.begin_time
            # Some annotations in IHM setting exceed audio duration, so we
            # ignore such segments
            if seg_info.end_time > recording.duration:
                logging.warning(
                    f"Segment {meetingid}-{channel}-{seg_idx} exceeds "
                    f"recording duration. Not adding to supervisions."
                )
                continue
            if duration > 0:
                segments.append(
                    SupervisionSegment(
                        id=f"{recording.id}-{seg_idx}",
                        recording_id=recording.id,
                        start=seg_info.begin_time,
                        duration=duration,
                        channel=0,
                        language="English",
                        speaker=seg_info.speaker,
                        gender=seg_info.gender,
                        text=seg_info.text,
                    )
                )
    supervision_set = SupervisionSet.from_segments(segments)
    validate_recordings_and_supervisions(recording_set, supervision_set)
    manifests = defaultdict(dict)
    dataset_parts = PARTITIONS[partition]
    for part in ["train", "dev", "test"]:
        # Get recordings for current data split
        recording_part = recording_set.filter(lambda x: x.id.split('.')[0] in dataset_parts[part])
        supervision_part = supervision_set.filter(
            lambda x: x.recording_id.split('.')[0] in dataset_parts[part]
        )

        # Write to output directory if a path is provided
        if output_dir is not None:
            recording_part.to_json(output_dir / f"recordings_{part}.json")
            supervision_part.to_json(output_dir / f"supervisions_{part}.json")
        validate_recordings_and_supervisions(recording_part, supervision_part)
        # Combine all manifests into one dictionary
        manifests[part] = {"recordings": recording_part, "supervisions": supervision_part}
    return dict(manifests)


def main():
    prepare_ami('/export/common/data/corpora/amicorpus/',
    '/exp/aarora/icefall_work_env/other/lhotse_output/ami')


if __name__ == '__main__':
    main()