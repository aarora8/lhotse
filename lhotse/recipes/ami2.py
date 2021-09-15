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

MEETINGS = {
    'EN2001': ['EN2001a', 'EN2001b', 'EN2001d', 'EN2001e'],
    'EN2002': ['EN2002a', 'EN2002b', 'EN2002c', 'EN2002d'],
    'EN2003': ['EN2003a'],
    'EN2004': ['EN2004a'],
    'EN2005': ['EN2005a'],
    'EN2006': ['EN2006a','EN2006b'],
    'EN2009': ['EN2009b','EN2009c','EN2009d'],
    'ES2002': ['ES2002a','ES2002b','ES2002c','ES2002d'],
    'ES2003': ['ES2003a','ES2003b','ES2003c','ES2003d'],
    'ES2004': ['ES2004a','ES2004b','ES2004c','ES2004d'],
    'ES2005': ['ES2005a','ES2005b','ES2005c','ES2005d'],
    'ES2006': ['ES2006a','ES2006b','ES2006c','ES2006d'],
    'ES2007': ['ES2007a','ES2007b','ES2007c','ES2007d'],
    'ES2008': ['ES2008a','ES2008b','ES2008c','ES2008d'],
    'ES2009': ['ES2009a','ES2009b','ES2009c','ES2009d'],
    'ES2010': ['ES2010a','ES2010b','ES2010c','ES2010d'],
    'ES2011': ['ES2011a','ES2011b','ES2011c','ES2011d'],
    'ES2012': ['ES2012a','ES2012b','ES2012c','ES2012d'],
    'ES2013': ['ES2013a','ES2013b','ES2013c','ES2013d'],
    'ES2014': ['ES2014a','ES2014b','ES2014c','ES2014d'],
    'ES2015': ['ES2015a','ES2015b','ES2015c','ES2015d'],
    'ES2016': ['ES2016a','ES2016b','ES2016c','ES2016d'],
    'IB4001': ['IB4001'],
    'IB4002': ['IB4002'],
    'IB4003': ['IB4003'],
    'IB4004': ['IB4004'],
    'IB4005': ['IB4005'],
    'IB4010': ['IB4010'],
    'IB4011': ['IB4011'],
    'IN1001': ['IN1001'],
    'IN1002': ['IN1002'],
    'IN1005': ['IN1005'],
    'IN1007': ['IN1007'],
    'IN1008': ['IN1008'],
    'IN1009': ['IN1009'],
    'IN1012': ['IN1012'],
    'IN1013': ['IN1013'],
    'IN1014': ['IN1014'],
    'IN1016': ['IN1016'],
    'IS1000': ['IS1000a','IS1000b','IS1000c','IS1000d'],
    'IS1001': ['IS1001a','IS1001b','IS1001c','IS1001d'],
    'IS1002': ['IS1002b','IS1002c','IS1002d'],
    'IS1003': ['IS1003a','IS1003b','IS1003c','IS1003d'],
    'IS1004': ['IS1004a','IS1004b','IS1004c','IS1004d'],
    'IS1005': ['IS1005a','IS1005b','IS1005c'],
    'IS1006': ['IS1006a','IS1006b','IS1006c','IS1006d'],
    'IS1007': ['IS1007a','IS1007b','IS1007c','IS1007d'],
    'IS1008': ['IS1008a','IS1008b','IS1008c','IS1008d'],
    'IS1009': ['IS1009a','IS1009b','IS1009c','IS1009d'],
    'TS3003': ['TS3003a','TS3003b','TS3003c','TS3003d'],
    'TS3004': ['TS3004a','TS3004b','TS3004c','TS3004d'],
    'TS3005': ['TS3005a','TS3005b','TS3005c','TS3005d'],
    'TS3006': ['TS3006a','TS3006b','TS3006c','TS3006d'],
    'TS3007': ['TS3007a','TS3007b','TS3007c','TS3007d'],
    'TS3008': ['TS3008a','TS3008b','TS3008c','TS3008d'],
    'TS3009': ['TS3009a','TS3009b','TS3009c','TS3009d'],
    'TS3010': ['TS3010a','TS3010b','TS3010c','TS3010d'],
    'TS3011': ['TS3011a','TS3011b','TS3011c','TS3011d'],
    'TS3012': ['TS3012a','TS3012b','TS3012c','TS3012d'],
}

PARTITIONS = {
    'full-corpus-asr': {
        'train': [meeting for session in [
                'ES2002','ES2005','ES2006','ES2007','ES2008','ES2009','ES2010','ES2012','ES2013',
                'ES2015','ES2016','IS1000','IS1001','IS1002','IS1003','IS1004','IS1005','IS1006',
                'IS1007','TS3005','TS3008','TS3009','TS3010','TS3011','TS3012','EN2001','EN2003',
                'EN2004','EN2005','EN2006','EN2009','IN1001','IN1002','IN1005','IN1007','IN1008',
                'IN1009','IN1012','IN1013','IN1014','IN1016','ES2014','TS3007','ES2003','TS3006'
            ] for meeting in MEETINGS[session]],
        'dev': [meeting for session in [
                'ES2011','IS1008','TS3004','IB4001','IB4002','IB4003','IB4004','IB4010','IB4011'
            ] for meeting in MEETINGS[session]],
        'test': [meeting for session in [
                'ES2004','IS1009','TS3003','EN2002'
            ] for meeting in MEETINGS[session]]
    }
}
MICS = ['ihm']


def download_ami(
    target_dir: Pathlike = ".",
    annotations_dir: Optional[Pathlike] = None,
    url: Optional[str] = "http://groups.inf.ed.ac.uk/ami",
    mic: Optional[str] = "ihm",
) -> None:
    """
    Download AMI audio and annotations for provided microphone setting.
    :param target_dir: Pathlike, the path to store the data.
    :param annotations_dir: Pathlike (default = None), path to save annotations zip file
    :param force_download: bool (default = False), if True, download even if file is present.
    :param url: str (default = 'http://groups.inf.ed.ac.uk/ami'), AMI download URL.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic setting.
    """
    target_dir = Path(target_dir)

    annotations_dir = target_dir if not annotations_dir else annotations_dir

    #? Annotations
    logging.info("Downloading AMI annotations")
    annotations_name = "annotations1.zip"
    annotations_path = annotations_dir / annotations_name
    annotations_url = f"{url}/AMICorpusAnnotations/ami_manual_annotations_v1.6.1_export.gzip"
    if not annotations_path.is_file():
        urllib.request.urlretrieve(annotations_url, filename=annotations_path)


class AmiSegmentAnnotation(NamedTuple):
    text: str
    speaker: str
    gender: str
    begin_time: Seconds
    end_time: Seconds


def parse_ami_annotations(
    annotations_zip: Pathlike, max_pause: float
) -> Dict[str, List[SupervisionSegment]]:
    encoding = 'utf-8'
    annotations = []
    with gzip.open(annotations_zip,'rb') as fin:        
        for line in fin:
            if line.startswith(b'Found'):
                continue
            if line.startswith(b'Obs'):
                continue
            if line.startswith(b'obs'):
                continue
            line = str(line, encoding)
            annotations.append(line)
    # annotations.append("ES2002a\tB\tFEE005 \t1 \t55.415\t77.456\t55.415\t77.29\tUm well this is the kick-off meeting for our our project . Um and um this is just what we're gonna be doing over the next twenty five minutes . Um so first of all , just to kind of make sure that we all know each other , I'm Laura and I'm the project manager . Do you want to introduce yourself again ? \t60.35 67.55 69.85 72.79 74.42 77.29 \t\n")
    # <meetingID> <skip> <spkr> <channel> <skip> <skip> <start time> <end time> <transcripts> <partial end time1> <partial end time2> <partial end time3>
    count=0
    for line in annotations:
        parts = line.strip().split('\t')
        meeting_id = parts[0]
        spkr = parts[2]
        channel_id = parts[3]
        st_time = float(parts[6])
        end_time = float(parts[7])
        transcript = parts[8]
        intermediate_endtimes = parts[9].split()

        transcript = transcript.strip().split('.')
        total_comma_counts = 0
        for index, transcript_part in enumerate(transcript):
            if not transcript_part:
                continue
            total_comma_counts = total_comma_counts + transcript_part.count(',')
            try: 
                float(intermediate_endtimes[index + total_comma_counts])
            except:
                print(transcript)
                print(intermediate_endtimes)
            # print(transcript_part)
            # print(intermediate_endtimes[index + total_comma_counts])
            count = count + 1
    print(count)

        
    # with zipfile.ZipFile(annotations_zip, "r") as archive:
    #     # First we get global speaker ids and channels
    #     global_spk_id = {}
    #     channel_id = {}
    #     with archive.open("corpusResources/meetings.xml") as f:
    #         tree = ET.parse(f)
    #         for meeting in tree.getroot():
    #             meet_id = meeting.attrib["observation"]
    #             for speaker in meeting:
    #                 local_id = (meet_id, speaker.attrib["nxt_agent"])
    #                 global_spk_id[local_id] = speaker.attrib["global_name"]
    #                 channel_id[local_id] = int(speaker.attrib["channel"])

    #     # Now iterate over all alignments
    #     for file in archive.namelist():
    #         if file.startswith("words/") and file[-1] != "/":
    #             meet_id, x, _, _ = file.split("/")[1].split(".")
    #             if (meet_id, x) not in global_spk_id:
    #                 logging.warning(
    #                     f"No speaker {meet_id}.{x} found! Skipping annotation."
    #                 )
    #                 continue
    #             spk = global_spk_id[(meet_id, x)]
    #             channel = channel_id[(meet_id, x)]
    #             tree = ET.parse(archive.open(file))
    #             key = (meet_id, spk, channel)
    #             if key not in annotations:
    #                 annotations[key] = []
    #             for child in tree.getroot():
    #                 # If the alignment does not contain start or end time info,
    #                 # ignore them. Also, only consider words in the alignment XML files.
    #                 if (
    #                     "starttime" not in child.attrib
    #                     or "endtime" not in child.attrib
    #                     or child.tag != "w"
    #                 ):
    #                     continue
    #                 text = child.text if child.tag == "w" else child.attrib["type"]
    #                 # to convert HTML escape sequences
    #                 text = html.unescape(text)
    #                 annotations[key].append(
    #                     AmiSegmentAnnotation(
    #                         text=text,
    #                         speaker=spk,
    #                         gender=spk[0],
    #                         begin_time=float(child.attrib["starttime"]),
    #                         end_time=float(child.attrib["endtime"]),
    #                     )
    #                 )

    #     # Post-process segments and combine neighboring segments from the same speaker
    #     for key in annotations:
    #         new_segs = []
    #         cur_seg = list(annotations[key])[0]
    #         for seg in list(annotations[key])[1:]:
    #             if seg.begin_time - cur_seg.end_time <= max_pause:
    #                 cur_seg = cur_seg._replace(
    #                     text=f"{cur_seg.text} {seg.text}", end_time=seg.end_time
    #                 )
    #             else:
    #                 new_segs.append(cur_seg)
    #                 cur_seg = seg
    #         new_segs.append(cur_seg)
    #         annotations[key] = new_segs
    #return annotations


def prepare_ami(
    data_dir: Pathlike,
    annotations_dir: Pathlike,
    output_dir: Pathlike,
    mic: Optional[str] = "ihm",
    partition: Optional[str] = "full-corpus-asr",
    max_pause: Optional[float] = 0.0,
) -> Dict[str, Dict[str, Union[RecordingSet, SupervisionSet]]]:
    """
    Returns the manifests which consist of the Recordings and Supervisions
    :param data_dir: Pathlike, the path of the data dir.
    :param output_dir: Pathlike, the path where to write the manifests.
    :param mic: str {'ihm','ihm-mix','sdm','mdm'}, type of mic to use.
    :param partition: str {'full-corpus','full-corpus-asr','scenario-only'}, AMI official data split
    :param max_pause: float (default = 0.0), max pause allowed between word segments to combine segments
    :return: a Dict whose key is ('train', 'dev', 'eval'), and the values are dicts of manifests under keys
        'recordings' and 'supervisions'.
    """
    data_dir = Path(data_dir)
    annotations_dir = Path(annotations_dir)
    assert data_dir.is_dir(), f"No such directory: {data_dir}"
    assert mic in MICS, f"Mic {mic} not supported"
    assert partition in PARTITIONS, f"Partition {partition} not supported"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Parsing AMI annotations")
    annotations = parse_ami_annotations(
        annotations_dir / "annotations1.zip", max_pause=max_pause
    )
    annotation_by_id_and_channel = {
        (key[0], key[2]): annotations[key] for key in annotations
    }
    #? Audio
    logging.info("Preparing recording manifests")
    wav_dir = data_dir
    audio_paths = (wav_dir.rglob("*Headset-?.wav"))
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
        sessionid = recording.id.split('.')[0]
        channel = int(channel.strip())
        sessionid = str(sessionid.strip())
        annotation = annotation_by_id_and_channel.get((sessionid, channel))
        if annotation is None:
            logging.warning(
            f"No annotation found for recording {sessionid} "
            f"(file {channel})")
            continue
        for seg_idx, seg_info in enumerate(annotation):
            duration = seg_info.end_time - seg_info.begin_time
            # Some annotations in IHM setting exceed audio duration, so we
            # ignore such segments
            if seg_info.end_time > recording.duration:
                logging.warning(
                    f"Segment {sessionid}-{channel}-{seg_idx} exceeds "
                    f"recording duration. Not adding to supervisions."
                )
                continue
            if duration > 0:
                segments.append(
                    SupervisionSegment(
                        id=f"{sessionid}.Headset-{channel}-{seg_idx}",
                        recording_id=f"{recording.id}",
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
    download_ami('/exp/aarora/icefall_work_env/other/lhotse_output/ami/')

    prepare_ami('/export/common/data/corpora/amicorpus/',
    '/exp/aarora/icefall_work_env/other/lhotse_output/ami',
    '/exp/aarora/icefall_work_env/other/lhotse_output/ami')


if __name__ == '__main__':
    main()