from distutils.spawn import spawn
import random
import blobfile as bf
from mpi4py import MPI
import numpy as np
import torch as th
import os
import pickle
import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets.patches_utils import PatchesClips
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from moviepy.editor import ThumbnailsFileClip


def load_data(
        *,
        data_dir,
        batch_size,
        patches_size,
        thumbnails_size,
        deterministic=False,
        random_flip=True,
        num_workers=0,
        patches_fps=10,
        thumbnails_fps=None,
        frame_gap=1,
        drop_last=True
):
    if not data_dir:
        raise ValueError("unspecified data directory")

    all_files = []

    all_files.extend(_list_patches_files_recursively(data_dir))
    if MPI.COMM_WORLD.Get_rank() == 0:
        print(f"len(data loader):{len(all_files)}")

    clip_length_in_frames = patches_size[0]
    frames_between_clips = 1
    meta_fname = os.path.join(data_dir,
                              f"patches_clip_f{clip_length_in_frames}_g{frames_between_clips}_r{patches_fps}.pkl")

    if not os.path.exists(meta_fname):
        if MPI.COMM_WORLD.Get_rank() == 0:
            print(f"prepare {meta_fname}...")

        patches_clips = PatchesClips(
            patches_paths=all_files,
            clip_length_in_frames=clip_length_in_frames,  # 64
            frames_between_clips=frames_between_clips,
            num_workers=16,
            frame_rate=patches_fps
        )

        if MPI.COMM_WORLD.Get_rank() == 0:
            with open(meta_fname, 'wb') as f:
                pickle.dump(patches_clips.metadata, f)

    else:
        print(f"load {meta_fname}...")
        metadata = pickle.load(open(meta_fname, 'rb'))

        patches_clips = PatchesClips(patches_paths=all_files,
                                     clip_length_in_frames=clip_length_in_frames,  # 64
                                     frames_between_clips=frames_between_clips,
                                     frame_rate=patches_fps,
                                     _precomputed_metadata=metadata)


    dataset = MultimodalDataset(
        patches_size=patches_size,
        thumbnails_size=thumbnails_size,
        patches_clips=patches_clips,
        shard=MPI.COMM_WORLD.Get_rank(),
        num_shards=MPI.COMM_WORLD.Get_size(),
        random_flip=random_flip,
        thumbnails_fps=thumbnails_fps,
        frame_gap=frame_gap
    )

    if deterministic:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=drop_last
        )
    else:
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=drop_last
        )

    while True:
        yield from loader


def _list_patches_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["avi", "gif", "mp4"]:

            results.append(full_path)
        elif bf.isdir(full_path):

            results.extend(_list_patches_files_recursively(full_path))
    return results


class Dual_diffusion_datasets(Dataset):

    def __init__(
            self,
            patches_size,
            thumbnails_size,
            patches_clips,
            shard=0,
            num_shards=1,
            random_flip=True,
            thumbnails_fps=None,
            frame_gap=1
    ):
        super().__init__()
        self.patches_size = patches_size  # [f,C,H,W]
        self.thumbnails_size = thumbnails_size  # [c,len]
        self.random_flip = random_flip
        self.patches_clips = patches_clips
        self.thumbnails_fps = thumbnails_fps
        self.frame_gap = frame_gap
        self.size = self.patches_clips.num_clips()
        self.shuffle_indices = [i for i in list(range(self.size))[shard:][::num_shards]]
        random.shuffle(self.shuffle_indices)

    def __len__(self):
        return len(self.shuffle_indices)

    def process_patches(self, patches):  # size:64:64

        patches = patches.permute([0, 3, 1, 2])
        old_size = patches.shape[2:4]
        ratio = min(float(self.patches_size[2]) / (old_size[0]), float(self.patches_size[3]) / (old_size[1]))
        new_size = tuple([int(i * ratio) for i in old_size])
        pad_w = self.patches_size[3] - new_size[1]
        pad_h = self.patches_size[2] - new_size[0]
        top, bottom = pad_h // 2, pad_h - (pad_h // 2)
        left, right = pad_w // 2, pad_w - (pad_w // 2)
        transform = T.Compose(
            [T.RandomHorizontalFlip(self.random_flip), T.Resize(new_size, interpolation=InterpolationMode.BICUBIC),
             T.Pad((left, top, right, bottom))])
        patches_new = transform(patches)
        return patches_new

    def get_item(self, idx):

        while True:
            try:
                patches, raw_thumbnails, info, patches_idx = self.patches_clips.get_clip(idx)
            except Exception:
                idx = (idx + 1) % self.patches_clips.num_clips()
                continue
            break

        if len(patches) < self.patches_size[0]:
            append = self.patches_size[0] - len(patches)
            patches = th.cat([patches, patches[-1:].repeat(append, 1, 1, 1)], dim=0)
        else:
            patches = patches[:self.patches_size[0]]

        patches_after_process = self.process_patches(patches)
        patches_after_process = patches_after_process.float() / 127.5 - 1  # 0-1

        patches_idx, clip_idx = self.patches_clips.get_clip_location(idx)
        duration_per_frame = self.patches_clips.patches_pts[patches_idx][1] - \
                             self.patches_clips.patches_pts[patches_idx][0]
        patches_fps = self.patches_clips.patches_fps[patches_idx]
        thumbnails_fps = self.thumbnails_fps if self.thumbnails_fps else info['thumbnails_fps']

        clip_pts = self.patches_clips.clips[patches_idx][clip_idx]
        clip_pid = clip_pts // duration_per_frame

        start_t = (clip_pid[0] / patches_fps * 1.).item()
        end_t = ((clip_pid[-1] + 1) / patches_fps * 1.).item()

        patches_path = self.patches_clips.patches_paths[patches_idx]
        raw_thumbnails = ThumbnailsFileClip(patches_path, fps=thumbnails_fps).subclip(start_t, end_t)

        thumbnails = np.zeros(self.thumbnails_size)
        raw_thumbnails = raw_thumbnails.to_soundarray()
        if raw_thumbnails.shape[1] == 2:
            raw_thumbnails = raw_thumbnails[:, 0:1].T  # pick one channel
        if raw_thumbnails.shape[1] < self.thumbnails_size[1]:
            thumbnails[:, :raw_thumbnails.shape[1]] = raw_thumbnails
        elif raw_thumbnails.shape[1] >= self.thumbnails_size[1]:
            thumbnails = raw_thumbnails[:, :self.thumbnails_size[1]]

        thumbnails = th.tensor(thumbnails)

        return patches_after_process, thumbnails

    def __getitem__(self, idx):
        idx = self.shuffle_indices[idx]
        patches_after_process, thumbnails = self.get_item(idx)

        return patches_after_process, thumbnails


if __name__ == '__main__':

    thumbnails_fps = 16000
    patches_fps = 10
    batch_size = 4
    seconds = 1.6
    image_resolution = 64

    dataset64 = load_data(
        data_dir="/data/test",
        batch_size=batch_size,
        patches_size=[int(seconds * patches_fps), 3, 64, 64],
        thumbnails_size=[1, int(seconds * thumbnails_fps)],
        frame_gap=1,
        random_flip=False,
        num_workers=0,
        deterministic=True,
        patches_fps=patches_fps,
        thumbnails_fps=thumbnails_fps
    )

    group = 0

    while True:
        group += 1
        batch_patches, batch_thumbnails, cond = next(dataset64)

