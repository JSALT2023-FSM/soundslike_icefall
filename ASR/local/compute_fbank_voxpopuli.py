#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# 	       2022  Xiaomi Crop.        (authors: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This file computes fbank features of the VoxPopuli dataset.
It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import logging
from pathlib import Path

import torch
import torchaudio
from lhotse import CutSet, KaldifeatFbank, KaldifeatFbankConfig
from lhotse.audio import set_ffmpeg_torchaudio_info_enabled
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
torch.multiprocessing.set_sharing_strategy("file_system")

# The following can speed up audio loading on some systems.
torchaudio.set_audio_backend("soundfile")
set_ffmpeg_torchaudio_info_enabled(False)


def compute_fbank_voxpopuli():
    src_dir = Path("data/manifests")
    feats_dir = Path("data/fbank")
    num_mel_bins = 80

    # number of workers in dataloader
    num_workers = 4

    # number of seconds in a batch
    batch_duration = 2000

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda", 0)
    extractor = KaldifeatFbank(KaldifeatFbankConfig(device=device))

    dataset_parts = ("train", "dev", "test")

    prefix = "voxpopuli-asr-en"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    for partition in dataset_parts:
        cuts_path = src_dir / f"cuts_{partition}.jsonl.gz"
        if cuts_path.is_file():
            logging.warning(f"{cuts_path} exists - skipping")
            continue

        cut_set = CutSet.from_manifests(**manifests[partition]).trim_to_supervisions(
            keep_overlapping=False
        )

        logging.info(f"Computing features for {partition}")

        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{feats_dir}/feats_{partition}",
            manifest_path=cuts_path,
            num_workers=num_workers,
            batch_duration=batch_duration,
            overwrite=True,
        )


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)

    compute_fbank_voxpopuli()
