#encoding=utf-8
# BSD 3- Clause License Copyright (c) 2023, Tecorigin Co., Ltd. All rights
# reserved.
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
# Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software
# without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY,OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)  ARISING IN ANY
# WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
# OF SUCH DAMAGE.
import os
import re
import numpy as np
import librosa
import paddle
import random
from tqdm import tqdm
from paddle.io import Dataset
from paddle.io import DataLoader
import utils_all as ut
import pandas as pd
import paddlefcpe


def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir) + 1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list

                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue

                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext) + 1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list


def get_data_loaders(args):
    wav2mel = paddlefcpe.spawn_wav2mel(args, device='cpu')
    data_train = F0Dataset(
        path_root=args.data.train_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sr,
        duration=args.data.duration,
        load_all_data=args.train.cache_all_data,
        whole_audio=False,
        extensions=args.data.extensions,
        device=args.train.cache_device,
        wav2mel=wav2mel,
        aug_noise=args.train.aug_noise,
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        aug_mask=args.train.aug_mask,
        aug_mask_v_o=args.train.aug_mask_v_o,
        aug_mask_vertical_factor=args.train.aug_mask_vertical_factor,
        aug_mask_vertical_factor_v_o=args.train.aug_mask_vertical_factor_v_o,
        aug_mask_iszeropad_mode=args.train.aug_mask_iszeropad_mode,
        aug_mask_block_num=args.train.aug_mask_block_num,
        aug_mask_block_num_v_o=args.train.aug_mask_block_num_v_o,
        aug_keyshift=args.train.aug_keyshift,
        keyshift_min=args.train.keyshift_min,
        keyshift_max=args.train.keyshift_max,
        f0_min=args.model.f0_min,
        f0_max=args.model.f0_max,
        f0_shift_mode='keyshift',
    )
    loader_train = DataLoader(
        data_train,
        batch_size=args.train.batch_size,
        num_workers=args.train.num_workers,
        use_shared_memory=False,
    )
    data_valid = F0Dataset(
        path_root=args.data.valid_path,
        waveform_sec=args.data.duration,
        hop_size=args.mel.hop_size,
        sample_rate=args.mel.sr,
        duration=args.data.duration,
        load_all_data=args.train.cache_all_data,
        whole_audio=True,
        extensions=args.data.extensions,
        wav2mel=wav2mel,
        aug_noise=args.train.aug_noise,
        noise_ratio=args.train.noise_ratio,
        brown_noise_ratio=args.train.brown_noise_ratio,
        aug_mask=args.train.aug_mask,
        aug_mask_v_o=args.train.aug_mask_v_o,
        aug_mask_vertical_factor=args.train.aug_mask_vertical_factor,
        aug_mask_vertical_factor_v_o=args.train.aug_mask_vertical_factor_v_o,
        aug_mask_iszeropad_mode=args.train.aug_mask_iszeropad_mode,
        aug_mask_block_num=args.train.aug_mask_block_num,
        aug_mask_block_num_v_o=args.train.aug_mask_block_num_v_o,
        aug_keyshift=False,
    )
    loader_valid = DataLoader(
        data_valid,
        batch_size=1,
        num_workers=0,
        use_shared_memory=False,
    )
    return loader_train, loader_valid


class F0Dataset(Dataset):
    def __init__(
            self,
            path_root,
            waveform_sec,
            hop_size,
            sample_rate,
            duration,
            load_all_data=True,
            whole_audio=False,
            extensions=['wav'],
            device='cpu',
            wav2mel=None,
            aug_noise=False,
            noise_ratio=0.7,
            brown_noise_ratio=1.,
            aug_mask=False,
            aug_mask_v_o=False,
            aug_mask_vertical_factor=0.05,
            aug_mask_vertical_factor_v_o=0.3,
            aug_mask_iszeropad_mode='randon',  # randon zero or noise
            aug_mask_block_num=1,
            aug_mask_block_num_v_o=4,
            aug_keyshift=True,
            keyshift_min=-5,
            keyshift_max=12,
            f0_min=32.70,
            f0_max=1975.5,
            f0_shift_mode='keyshift',
            snb_noise=None,
            noise_beta=0,
    ):
        super().__init__()
        self.music_spk_id = 1
        self.wav2mel = wav2mel
        self.waveform_sec = waveform_sec
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.path_root = path_root
        self.duration = duration
        self.aug_noise = aug_noise
        self.noise_ratio = noise_ratio
        self.brown_noise_ratio = brown_noise_ratio
        self.aug_mask = aug_mask
        self.aug_mask_v_o = aug_mask_v_o
        self.aug_mask_vertical_factor = aug_mask_vertical_factor
        self.aug_mask_vertical_factor_v_o = aug_mask_vertical_factor_v_o
        self.aug_mask_iszeropad_mode = aug_mask_iszeropad_mode
        self.aug_mask_block_num = aug_mask_block_num
        self.aug_mask_block_num_v_o = aug_mask_block_num_v_o
        self.aug_keyshift = aug_keyshift
        self.keyshift_min = keyshift_min
        self.keyshift_max = keyshift_max
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.f0_shift_mode = f0_shift_mode
        self.device = device
        self.load_all_data = load_all_data
        self.snb_noise = snb_noise
        self.noise_beta = noise_beta

        self.paths = traverse_dir(
            os.path.join(path_root, 'audio'),
            extensions=extensions,
            is_pure=True,
            is_sort=True,
            is_ext=True
        )

        self.whole_audio = whole_audio
        self.data_buffer = {}
        self.device = device
        if load_all_data:
            print('Load all the data from :', path_root)
        else:
            print('Load the f0, volume data from :', path_root)

        with paddle.no_grad():
            tasks = [self.paths]
            for data_buffer in map(self.load_data, tasks):
                self.data_buffer.update(data_buffer)

            self.paths = np.array(self.paths, dtype=object)
            self.data_buffer = pd.DataFrame(self.data_buffer)

    def load_data(self, paths):
        with paddle.no_grad():
            data_buffer = {}
            for name_ext in tqdm(paths):
                path_audio = os.path.join(self.path_root, 'audio', name_ext)
                duration = librosa.get_duration(filename=path_audio, sr=self.sample_rate)

                path_f0 = os.path.join(self.path_root, 'f0', name_ext) + '.npy'
                f0 = np.load(path_f0)[:, None]
                # f0 = torch.from_numpy(f0).float().unsqueeze(-1).to(self.device)

                r'''
                if self.n_spk is not None and self.n_spk > 1:
                    dirname_split = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                    t_spk_id = spk_id = int(dirname_split) if str.isdigit(dirname_split) else 0
                    if spk_id < 1 or spk_id > self.n_spk:
                        raise ValueError(
                            ' [x] Muiti-speaker traing error : spk_id must be a positive integer from 1 to n_spk ')
                else:
                    pass
                '''
                    # spk_id = 1
                    # t_spk_id = spk_id
                # spk_id = torch.LongTensor(np.array([spk_id])).to(self.device)
                # spk_id = np.array([spk_id])

                if self.load_all_data:
                    audio, sr = librosa.load(path_audio, sr=self.sample_rate)
                    if len(audio.shape) > 1:
                        audio = librosa.to_mono(audio)
                    # audio = torch.from_numpy(audio).to(device)

                    # path_audio = os.path.join(self.path_root, 'npaudiodir', name_ext) + '.npy'
                    # audio = np.load(path_audio)

                    if False:#spk_id == self.music_spk_id:
                        path_music = os.path.join(self.path_root, 'music', name_ext)# + '.npy'
                        # audio_music = np.load(path_music)
                        audio_music, _ = librosa.load(path_music, sr=self.sample_rate)
                        if len(audio_music.shape) > 1:
                            audio_music = librosa.to_mono(audio_music)
                    else:
                        audio_music = None

                    """
                    data_buffer[name_ext] = {
                        'duration': duration,
                        'audio': audio,
                        'f0': f0,
                        'spk_id': spk_id,
                        't_spk_id': t_spk_id,
                    }
                    """
                    data_buffer[name_ext] = (duration, f0, audio, None)
                else:
                    '''
                    if spk_id == self.music_spk_id:
                        use_music = True
                    else:
                        use_music = None

                    data_buffer[name_ext] = {
                        'duration': duration,
                        'f0': f0,
                        'spk_id': spk_id,
                        't_spk_id': t_spk_id
                    }
                    '''
                    data_buffer[name_ext] = (duration, f0, None)
            return data_buffer

    def __getitem__(self, file_idx):
        with paddle.no_grad():
            name_ext = self.paths[file_idx]
            data_buffer = self.data_buffer[name_ext]
            # check duration. if too short, then skip
            if data_buffer[0] < (self.waveform_sec + 0.1):
                return self.__getitem__((file_idx + 1) % len(self.paths))

            # get item
            return self.get_data(name_ext, data_buffer)

    def get_data(self, name_ext, data_buffer):
        with paddle.no_grad():
            name = os.path.splitext(name_ext)[0]
            frame_resolution = self.hop_size / self.sample_rate
            duration = data_buffer[0]
            waveform_sec = duration if self.whole_audio else self.waveform_sec

            # load audio
            idx_from = 0 if self.whole_audio else random.uniform(0, duration - waveform_sec - 0.1)
            start_frame = int(idx_from / frame_resolution)
            units_frame_len = int(waveform_sec / frame_resolution)

            # load f0
            f0 = data_buffer[1].copy()
            f0 = paddle.to_tensor(f0,dtype="float32",place=paddle.CPUPlace())

            # load mel
            # audio = data_buffer.get('audio')
            if len(data_buffer) == 3:
                #path_audio = os.path.join(self.path_root, 'npaudiodir', name_ext) + '.npy'
                #audio = np.load(path_audio)
                path_audio = os.path.join(self.path_root, 'audio', name_ext)
                audio, _ = librosa.load(path_audio, sr=self.sample_rate)
                if len(audio.shape) > 1:
                    audio = librosa.to_mono(audio)
                if random.choice((False, True)) and (data_buffer[2] is not None):
                    path_music = os.path.join(self.path_root, 'music', name_ext)
                    audio_music, _ = librosa.load(path_music, sr=self.sample_rate)
                    if len(audio_music.shape) > 1:
                        audio_music = librosa.to_mono(audio_music)
                        audio = audio + audio_music
                        del audio_music
                        audio = 0.98 * audio / (np.abs(audio).max())

            else:
                audio = data_buffer[2].copy()
                if len(data_buffer) == 4:
                    if data_buffer[3] is not None:
                        if random.choice((False, True)):
                            audio_music = data_buffer[3].copy()
                            audio = audio + audio_music
                            del audio_music
                            audio = 0.98 * audio / (np.abs(audio).max())

            if random.choice((False, True)) and self.aug_keyshift:
                if self.f0_shift_mode == 'keyshift':
                    _f0_shift_mode = 'keyshift'
                elif self.f0_shift_mode == 'automax':
                    _f0_shift_mode = 'automax'
                elif self.f0_shift_mode == 'random':
                    _f0_shift_mode = random.choice(('keyshift', 'automax'))
                else:
                    raise ValueError('f0_shift_mode must be keyshift, automax or random')

                if _f0_shift_mode == 'keyshift':
                    keyshift = random.uniform(self.keyshift_min, self.keyshift_max)
                elif _f0_shift_mode == 'automax':
                    keyshift_max = 12 * np.log2(self.f0_max / f0.max)
                    keyshift_min = 12 * np.log2(self.f0_min / f0.min)
                    keyshift = random.uniform(keyshift_min, keyshift_max)
                with paddle.no_grad():
                    f0 = 2 ** (keyshift / 12) * f0
            else:
                keyshift = 0

            is_aug_noise = bool(random.randint(0, 1))

            if self.snb_noise is not None:
                audio = ut.add_noise_snb(audio, self.snb_noise, self.noise_beta)

            if self.aug_noise and is_aug_noise:
                if bool(random.randint(0, 1)):
                    audio = ut.add_noise(audio, noise_ratio=self.noise_ratio)
                else:
                    audio = ut.add_noise_slice(audio, self.sample_rate, self.duration, noise_ratio=self.noise_ratio,
                                               brown_noise_ratio=self.brown_noise_ratio)

            peak = np.abs(audio).max()
            audio = 0.98 * audio / peak
            audio = paddle.to_tensor(audio,dtype=paddle.float32,place=paddle.CPUPlace()).unsqueeze(0)
            with paddle.no_grad():
                mel = self.wav2mel(audio, sample_rate=self.sample_rate, keyshift=keyshift, no_cache_window=True).squeeze(0).cpu()

            if self.aug_mask and bool(random.randint(0, 1)) and not is_aug_noise:
                v_o = bool(random.randint(0, 1)) and self.aug_mask_v_o
                mel = mel.transpose([1, 0]) # mel = mel.transpose(-1, -2)
                if self.aug_mask_iszeropad_mode == 'zero':
                    iszeropad = True
                elif self.aug_mask_iszeropad_mode == 'noise':
                    iszeropad = False
                else:
                    iszeropad = bool(random.randint(0, 1))
                mel = ut.add_mel_mask_slice(mel, self.sample_rate, self.duration, hop_size=self.hop_size,
                                            vertical_factor=self.aug_mask_vertical_factor_v_o if v_o else self.aug_mask_vertical_factor,
                                            vertical_offset=v_o, iszeropad=iszeropad,
                                            block_num=self.aug_mask_block_num_v_o if v_o else self.aug_mask_block_num)
                mel = mel.transpose([1, 0]) # mel = mel.transpose(-1, -2)

            mel = mel[start_frame: start_frame + units_frame_len].detach()

            f0_frames = f0[start_frame: start_frame + units_frame_len].detach()

            # load spk_id
            # spk_id = data_buffer.get('spk_id')
            # spk_id = torch.LongTensor(spk_id).to(self.device)


            del audio
            # return dict(mel=mel, f0=f0_frames, spk_id=spk_id, name=name, name_ext=name_ext)
            output = (mel, f0_frames, name, name_ext)
            return output

    def __len__(self):
        return len(self.paths)
