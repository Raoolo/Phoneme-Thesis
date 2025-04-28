"""Paralinguistic Speech Explainer module"""
from venv import create

import numpy as np
from pydub import AudioSegment

from speechxai.explainers.visualization import plot_phoneme_importance_bar, plot_perturbation_trends, \
    plot_phoneme_importance_heatmap
from speechxai.utils import pydub_to_np, print_log
from speechxai.explainers.utils_removal import transcribe_audio
from IPython.display import display
# from pyrubberband.pyrb import time_stretch
from speechxai.explainers.explanation_speech import ExplanationSpeech
from audiomentations import (
    Compose,
    TimeStretch,
    PitchShift,
    RoomSimulator,
    AddBackgroundNoise,
    PolarityInversion,
    BitCrush,
    Gain,
)
import pandas as pd
import os

# If True, We use the audiostretchy library to perform time stretching
USE_AUDIOSTRETCH = True

# If True, We use the add_noise of torch audio
USE_ADD_NOISE_TORCHAUDIO = False
REFERENCE_STR = "-"


def _tmp_log1(
    verbose_target,
    original_gt,
    modified_trg,
    n_labels,
):
    # Log the original and modified predictions
    if n_labels > 1:
        print_log("Target label: ", verbose_target)
        print_log("gt", original_gt[verbose_target])
        print_log("m", modified_trg[verbose_target])

    else:
        print_log("gt", original_gt)
        print_log("m", modified_trg)


def _tmp_log2(
    verbose_target,
    original_gt,
    modified_trg,
    n_labels,
):
    # Log the differences between the original and modified predictions
    if n_labels > 1:
        print_log(
            [
                original_gt[verbose_target] - modified_trg[verbose_target][i]
                for i in range(modified_trg[verbose_target].shape[0])
            ]
        )

    else:
        print_log([original_gt - modified_trg[i] for i in range(modified_trg.shape[0])])


class ParalinguisticSpeechExplainer:
    # Used to explain the importance of paralinguistic features in speech audios
    # Perturbs the audio by applying various transformations and analyzes the effects on the predictions

    NAME = "paralinguistic_explainer_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def augmentation(
        self,
        perturbation_type: str,
        perturbation_value: float,
    ) -> Compose:
        """
        Creating a list of augmentations for each perturb_paraling type
        """
        if "pitch shifting" in perturbation_type:
            augment = Compose(
                [
                    PitchShift(
                        min_semitones=perturbation_value,
                        max_semitones=perturbation_value,
                        p=1.0,
                    ),
                ]
            )
        elif "time stretching" in perturbation_type:
            augment = Compose(
                [
                    TimeStretch(
                        min_rate=perturbation_value,
                        max_rate=perturbation_value,
                        p=1.0,
                        leave_length_unchanged=False,
                    ),
                ]
            )
        elif perturbation_type == "intensity":
            return Compose([
                Gain(
                    min_gain_db=perturbation_value,
                    max_gain_db=perturbation_value,
                    p=1.0
                )
            ])
        elif perturbation_type == "stress":
            # Combine pitch and intensity
            return Compose([
                TimeStretch(
                    min_rate=perturbation_value,
                    max_rate=perturbation_value,
                    p=1.0
                ),
                PitchShift(
                    min_semitones=perturbation_value,
                    max_semitones=perturbation_value,
                    p=1.0
                ),
                Gain(
                    min_gain_in_db=perturbation_value / 2,
                    max_gain_in_db=perturbation_value / 2,
                    p=1.0
                )
            ])
        elif perturbation_type == "reverberation":
            augment = Compose(
                [
                    RoomSimulator(
                        # room coordinates in meters
                        min_size_x=perturbation_value,
                        max_size_x=perturbation_value,
                        min_size_y=perturbation_value,
                        max_size_y=perturbation_value,
                        min_size_z=perturbation_value,
                        max_size_z=perturbation_value,
                        padding=0.5,
                        min_absorption_value=0.15,  # from doc, office/library, factory = 0.05
                        max_absorption_value=0.15,  # from doc, office/library
                        p=1.0,
                    ),
                ]
            )
        elif perturbation_type == "white noise" or perturbation_type == "noise":
            augment = Compose(
                [
                    AddBackgroundNoise(
                        sounds_path='/kaggle/working/speech/speechxai/explainers/white_noise.mp3',
                        # sounds_path=os.path.join(
                        #     os.path.dirname(__file__), "white_noise.mp3"),
                        min_snr_db=perturbation_value,
                        max_snr_db=perturbation_value,
                        noise_transform=PolarityInversion(),
                        p=1.0,
                    )
                ]
            )
        elif perturbation_type == "pink noise":
            augment = Compose(
                [
                    AddBackgroundNoise(
                        sounds_path='/kaggle/working/speech/speechxai/explainers/pink_noise.mp3',
                        # sounds_path=os.path.join(
                        #     os.path.dirname(__file__), "white_noise.mp3"),
                        min_snr_db=perturbation_value,
                        max_snr_db=perturbation_value,
                        noise_transform=PolarityInversion(),
                        p=1.0,
                    )
                ]
            )
        elif perturbation_type == "claps":
            augment = Compose(
                [
                    AddBackgroundNoise(
                        sounds_path='/kaggle/working/speech/speechxai/explainers/hand_clapping.mp3',
                        # sounds_path=os.path.join(
                        #     os.path.dirname(__file__), "white_noise.mp3"),
                        min_snr_db=perturbation_value,
                        max_snr_db=perturbation_value,
                        noise_transform=PolarityInversion(),
                        p=1.0,
                    )
                ]
            )
        elif perturbation_type == "degradation":
            augment = Compose(
                [
                    BitCrush(
                        min_bit_depth=perturbation_value,
                        max_bit_depth=perturbation_value,
                        p=1.0)
                ]
            )
        return augment

    def time_stretching_augmentation(
        self, audio_as: AudioSegment, perturbation_value: float
    ):
        # Apply time stretch to the audio with pydub
        # Perturbation value commands whether to slow down or speed up
        # Cannot be used for phonemes (too short)
        import audio_effects

        if perturbation_value < 1:  # slow down
            perturbed_audio_as = audio_effects.speed_down(audio_as, perturbation_value)
        else:   # speed up
            perturbed_audio_as = audio_as.speedup(perturbation_value)
        perturbed_audio, _ = pydub_to_np(perturbed_audio_as)
        return perturbed_audio.squeeze()

    def time_stretching_augmentation_AudioStretch(
        self, audio_path: str, perturbation_value: float
    ):
        # Apply time stretch to the audio with audiostretchy
        # Only used if parameter USE_AUDIOSTRETCH is true
        # Cannot be used for phonemes

        from audiostretchy.stretch import AudioStretch

        audio_stretch = AudioStretch()
        audio_stretch.open(audio_path)
        audio_stretch.stretch(ratio=perturbation_value)
        perturbated_audio_samples = np.array(audio_stretch.samples, dtype=np.float32)
        return perturbated_audio_samples

    def pitch_shifting_augmentation(
        self, audio_as: AudioSegment, perturbation_value: float
    ):
        """Apply pitch shift using pydub
        Perturbation value represents the number of octaves to shift the pitch"""

        # perturbation_value = octaves

        new_sample_rate = int(audio_as.frame_rate * (2.0**perturbation_value))

        perturbed_audio_as = audio_as._spawn(
            audio_as.raw_data, overrides={"frame_rate": new_sample_rate}
        )
        perturbed_audio_as = perturbed_audio_as.set_frame_rate(audio_as.frame_rate)
        perturbed_audio, _ = pydub_to_np(perturbed_audio_as)

        return perturbed_audio.squeeze()

    def add_white_noise_torchaudio(self, original_speech, noise_rate, perturbation_type):
        """
        Adds white noise using torchaudio
        Args:
        original_speech: np.array of shape (1, n_samples)
        noise_rate: signal-to-noise ratios in dB
        """

        import torchaudio.functional as F
        from copy import deepcopy
        import torch

        if perturbation_type == "white noise" or perturbation_type == "noise":
            WHITE_NOISE = os.path.join(os.path.dirname(__file__), "white_noise.mp3")
        elif perturbation_type == "pink noise":
            WHITE_NOISE = os.path.join(os.path.dirname(__file__), "pink_noise.mp3")
        elif perturbation_type == "claps":
            WHITE_NOISE = os.path.join(os.path.dirname(__file__), "hand_clapping.mp3")

        noise_as = AudioSegment.from_mp3(WHITE_NOISE)
        noise, frame_rate = pydub_to_np(noise_as)

        # Reshape and convert to torch tensor
        original_speech = torch.tensor(original_speech.reshape(1, -1))
        # Reshape and convert to torch tensor
        noise = torch.tensor(noise.reshape(1, -1))

        def extend_noise(noise, desired_length):
            """Extend the noise to match the length of the original audio"""
            noise_new = deepcopy(noise)
            while noise_new.shape[1] < desired_length:
                noise_new = torch.concat([noise_new[0], noise[0]]).reshape(1, -1)
            noise_new = noise_new[:, :desired_length]
            return noise_new

        noise_eq_length = extend_noise(noise, original_speech.shape[1])
        snr_dbs = torch.tensor([noise_rate, 10, 3])
        noisy_speeches = F.add_noise(original_speech, noise_eq_length, snr_dbs)
        noisy_speech = noisy_speeches[0:1].numpy()
        return noisy_speech

    def change_pitch_torchaudio(self, original_speech, frame_rate, perturbation_value):
        """
        Change pitch of audio using torchaudio
        The perturbation value represents the number of steps to shift the pitch
        Args:
        original_speech: np.array of shape (1, n_samples)
        perturbation_value:
        """

        import torchaudio.functional as F
        import torch

        # Reshape and convert to torch tensor
        audio_t = torch.tensor(original_speech.reshape(1, -1))
        perturbated_audio = F.pitch_shift(
            audio_t, frame_rate, n_steps=perturbation_value
        )
        perturbated_audio = perturbated_audio.numpy()
        return perturbated_audio

    def perturb_waveform(
        self,
        audio_path: str,
        perturbation_type: str,
        return_perturbations=False,
        verbose: bool = False,
        verbose_target: int = 0,
        perturbations_list: list = None,
    ):  # -> List[np.ndarray]:
        """
        Perturbate audio using pydub, by adding:
        - pitch shifting
        - time stretching
        - reverberation
        - noise
        """

        ## Load audio as pydub.AudioSegment
        audio_as = AudioSegment.from_wav(audio_path)
        audio, frame_rate = pydub_to_np(audio_as)

        ## Perturbate audio
        perturbed_audios = []

        if perturbations_list is not None:
            perturbations = perturbations_list
        else:
            if perturbation_type == "pitch shifting" or perturbation_type == "intensity":
                # perturbations = [-3.5, -2.5, -2, 2, 2.5, 3.5]
                # OK v2
                # perturbations = [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3]
                perturbations = np.arange(-5, 5.5, 0.5)
            elif perturbation_type == "pitch shifting down":
                # perturbations = [-3.5, -2.5, -2]
                # OK v2
                # perturbations = [-0.3, -0.2, -0.1]
                perturbations = np.arange(-5, 0, 0.5)
            elif perturbation_type == "pitch shifting up":
                # perturbations = [2, 2.5, 3.5]
                # ok v2
                # perturbations = [0.1, 0.2, 0.3]
                perturbations = np.arange(0.5, 5.5, 0.5)
            elif perturbation_type == "time stretching":
                # perturbations = [0.75, 0.80, 0.85, 1.15, 1.20, 1.25]
                # perturbations = [0.75, 0.85, 0.9, 1.15, 1.25, 1.35, 1.5]
                perturbations = [
                    #
                    0.65,
                    0.7,
                    0.75,
                    0.8,
                    0.85,
                    0.9,
                    0.95,
                    1.05,
                    1.10,
                    1.15,
                    1.2,
                    1.25,
                    1.3,
                    1.35,
                ]
                if USE_AUDIOSTRETCH:
                    perturbations = [0.55, 0.6] + perturbations + [1.4, 1.45]
            elif perturbation_type == "time stretching down":
                # perturbations = [1.15, 1.20, 1.25]
                if USE_AUDIOSTRETCH:
                    # For audio stretch it is the contrary...
                    perturbations = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
                else:
                    perturbations = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.35]
            elif perturbation_type == "time stretching up":
                if USE_AUDIOSTRETCH:
                    # For audio stretch it is the contrary...
                    perturbations = [1.05, 1.10, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45]
                else:
                    perturbations = [0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
            elif perturbation_type == "reverberation":
                perturbations = [3, 4, 5, 6, 7]
            elif "noise" in perturbation_type or "claps" in perturbation_type:
                perturbations = [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]
                if USE_ADD_NOISE_TORCHAUDIO:
                    perturbations = [40, 20, 10, 7.5, 5, 4, 3, 2, 1, 0.5, 0.1]
            else:
                raise ValueError(f"Perturbation '{perturbation_type}' is not available")

        if verbose:
            from IPython.display import Audio

            print_log("Original audio")
            # Display the original audio and show its info for a single class
            self._tmp_log_show_info(
                "Original audio",
                "",
                audio.squeeze(),
                verbose_target,
            )

        for perturbation_value in perturbations:
            if "time stretching" in perturbation_type:
                if USE_AUDIOSTRETCH:
                    perturbed_audio = self.time_stretching_augmentation_AudioStretch(
                        audio_path, perturbation_value
                    )
                else:
                    perturbed_audio = self.time_stretching_augmentation(
                        audio_as, perturbation_value
                    )
            elif "pitch shifting" in perturbation_type:
                # perturbed_audio = self.pitch_shifting_augmentation(
                #    audio_as, perturbation_value
                # )
                perturbed_audio = self.change_pitch_torchaudio(
                    audio, frame_rate, perturbation_value
                )
            elif "noise" in perturbation_type and USE_ADD_NOISE_TORCHAUDIO:
                perturbed_audio = self.add_white_noise_torchaudio(
                    audio, perturbation_value, perturbation_type
                )
            else:
                augment = self.augmentation(
                    perturbation_value=perturbation_value,
                    perturbation_type=perturbation_type,
                )
                perturbed_audio = augment(
                    samples=audio.squeeze(), sample_rate=frame_rate
                )

            if verbose:
                # Display the perturbed audio and show its info for a single class
                self._tmp_log_show_info(
                    perturbation_type,
                    perturbation_value,
                    perturbed_audio,
                    verbose_target,
                )
            perturbed_audios.append(perturbed_audio)

        if return_perturbations:
            return perturbed_audios, perturbations
        else:
            return perturbed_audios

    def perturb_waveform_words(
        self,
        audio_path: str,
        perturbation_type: str,
        return_perturbations=False,
        verbose: bool = False,
        verbose_target: int = 0,
        phonemization: bool = False,
        return_words: bool = False,
        single_perturbation_value: float = None,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        perturbations_list: list = None,
        sliding: bool = False,
    ):
        """
        Perturbate audio using pydub, by adding:
        - pitch shifting
        - time stretching
        - reverberation
        - noise
        """

        ## Load audio as pydub.AudioSegment
        audio_as = AudioSegment.from_wav(audio_path)
        audio, frame_rate = pydub_to_np(audio_as)

        # print(f"Audio channels: {audio.shape}")     # debug print

        _, words = transcribe_audio(    # using words for simplicity, but this contains phoneme in this case
            audio_path=audio_path,
            device=self.model_helper.device.type,
            batch_size=2,
            compute_type="float32",
            language=self.model_helper.language,
            phonemization=phonemization,  # word-level transcription
            window_size=window_size,
            respect_word_boundaries=respect_word_boundaries,
            sliding=sliding,
        )

        # define perturbations values, in the case of words/phonemes we might use a single value
        # both to avoid too many results and because we can isolate better word/phoneme effect
        if single_perturbation_value is not None:
            perturbations = [single_perturbation_value]
        elif perturbations_list is not None:
            perturbations = perturbations_list
        else:
            if perturbation_type == "pitch shifting" or perturbation_type == "intensity":
                perturbations = np.arange(-10, 11, 1)    # from doc [-12 to 12]
            elif perturbation_type == "pitch shifting down":
                perturbations = np.arange(-10, 0.5, 0.5)
            elif perturbation_type == "pitch shifting up":
                perturbations = np.arange(0.5, 10.5, 0.5)
            elif perturbation_type == "time stretching": # from 0 up
                perturbations = np.arange(0.25, 2.25, 0.25)
            elif perturbation_type == "time stretching down":
                perturbations = np.arange(0.2, 1.1, 0.1)
            elif perturbation_type == "time stretching up":
                perturbations = np.arange(1, 2.6, 0.1)
            elif perturbation_type == "reverberation":
                # perturbation is equivalent to the meters of a room (in this case it will be a cube)
                perturbations = np.arange(1, 6, 1)
            elif "noise" in perturbation_type or "claps" in perturbation_type:
                perturbations = np.arange(-30, 30, 5)
            elif perturbation_type == "degradation":
                perturbations = np.arange(1, 6, 1)
            else:
                raise ValueError(f"Perturbation '{perturbation_type}' is not available")

        if verbose:
            from IPython.display import Audio
            print_log("Original audio")
            # Display the original audio and show its info for a single class
            self._tmp_log_show_info(
                "Original audio",
                "",
                audio.squeeze(),
                verbose_target,
            )

        # Perturbate audio
        perturbed_audios = []

        # Iterate over each word/phoneme and apply perturbations
        for word in words:
            start_ms = word["start"] * 1000  # convert to milliseconds
            end_ms = word["end"] * 1000

            # extract the word segment
            word_segment = audio_as[start_ms:end_ms]
            word_audio, _ = pydub_to_np(word_segment)
            # print(f"len before perturbation: {len(word_audio)}")

            for perturbation_value in perturbations:
                if "time stretching" in perturbation_type:
                    # audiostretchy works on the full audio instead of segments so we don't use it
                    # the following also doesn't work for phonemes, they are too short
                        # Method 0 with audio_effects
                    # perturbed_word = self.time_stretching_augmentation(word_segment, perturbation_value)
                        # Method 1 with audiomentations (works but very low scores?, also seems to do opposite of value)
                    augment = self.augmentation(perturbation_value=perturbation_value,
                                                 perturbation_type=perturbation_type)
                    perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)
                        # Method 2 with rubberband (comparable results of M1)
                    # perturbed_word = self.time_stretch_phoneme(word_audio, perturbation_value, sample_rate=frame_rate)
                elif "pitch shifting" in perturbation_type:
                        # Method 1
                    # perturbed_word = self.change_pitch_torchaudio(word_audio, frame_rate, perturbation_value)
                        # Method 2 with audiomentations, MUCH faster and comparable results for multiple perturbations
                    augment = self.augmentation(perturbation_value=perturbation_value,
                                                perturbation_type=perturbation_type)
                    perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)
                elif perturbation_type == "noise" and USE_ADD_NOISE_TORCHAUDIO:
                    # barely any difference from audiomentations
                    perturbed_word = self.add_white_noise_torchaudio(word_audio, perturbation_value)
                else:
                    augment = self.augmentation(perturbation_value=perturbation_value, perturbation_type=perturbation_type)
                    perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)

                # debug print
                # print(f"Perturbed_words shape: {perturbed_word.shape}")
                # Reshape to match the expected shape (n_samples, 1)
                perturbed_word = perturbed_word.reshape(-1, 1)

                # print(f"len after perturbation: {len(perturbed_word)}")

                # Reconstruct the audio with the perturbed word/phoneme
                perturbed_audio = np.concatenate([
                    audio[:int(start_ms * frame_rate / 1000)],  # audio before the phoneme
                    perturbed_word,  # perturbed word/phoneme
                    audio[int(end_ms * frame_rate / 1000):]  # audio after the phoneme
                ])

                if verbose:
                    # Display the perturbed audio and show its info for a single class
                    self._tmp_log_show_info(
                        perturbation_type,
                        perturbation_value,
                        perturbed_audio,
                        verbose_target,
                    )

                # print(f"Original audio stats - mean: {audio.mean()}, std: {audio.std()}")
                # print(f"Perturbed audio stats - mean: {perturbed_audio.mean()}, std: {perturbed_audio.std()}")

                # plot spectrograms to visually verify the perturbations

                # import matplotlib.pyplot as plt
                # plt.figure(figsize=(10, 4))
                # plt.subplot(121)
                # plt.specgram(audio.squeeze(), Fs=16000)
                # plt.title("Original")
                # plt.subplot(122)
                # plt.specgram(perturbed_audio.squeeze(), Fs=16000)
                # plt.title(f"Perturbed {perturbation_type}")
                # plt.show()

                perturbed_audios.append(perturbed_audio)

        if return_perturbations:
            return perturbed_audios, perturbations
        elif return_words:
            return perturbed_audios, words, perturbations
        else:
            return perturbed_audios

    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        perturbation_type: str = None,
        verbose: bool = False,
        verbose_target: int = 0,
        word_level: bool = False,
        phonemization: bool = False,
        single_perturbation_value: float = None,
        complete_perturbation: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        perturbations_list: list = None,
        sliding: bool = False,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each feature (words/phoneme/paralinguistic) in the audio.
        Perturbs the audio and compare the model's predictions with the original predictions.
        """

        if word_level or phonemization and single_perturbation_value is not None:
            perturbed_audios, words, perturbations = self.perturb_waveform_words(
                audio_path=audio_path,
                perturbation_type=perturbation_type,
                return_words=True,
                verbose=verbose,
                verbose_target=verbose_target,
                perturbations_list=perturbations_list,
                phonemization=phonemization,
                single_perturbation_value=single_perturbation_value,
                window_size=window_size,
                respect_word_boundaries=respect_word_boundaries,
                sliding=sliding,
            )
        elif word_level or phonemization and single_perturbation_value is None:
            perturbed_audios, words, perturbations = self.perturb_waveform_words(
                audio_path=audio_path,
                perturbation_type=perturbation_type,
                return_words=True,
                verbose=verbose,
                verbose_target=verbose_target,
                perturbations_list=perturbations_list,
                phonemization=phonemization,
                window_size=window_size,
                respect_word_boundaries=respect_word_boundaries,
            )
        else:
            perturbed_audios = self.perturb_waveform(
                audio_path=audio_path,
                perturbation_type=perturbation_type,
                verbose=verbose,
                verbose_target=verbose_target,
                perturbations_list=perturbations_list,
            )

        # Get logits for modified and original audio for each class
        logits_modified = self.model_helper.predict(perturbed_audios)
        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]
        logits_original = self.model_helper.predict([audio])

        # Check if single label or multilabel scenario as for FSC
        n_labels = self.model_helper.n_labels

        if target_class is not None:
            targets = target_class
        else:
            if n_labels > 1:
                # Multilabel scenario as for FSC
                targets = [np.argmax(logits_original[i], axis=1)[0] for i in range(n_labels)]
            else:
                targets = np.argmax(logits_original, axis=1)[0]

        # Get the most important word/phoneme for each label
        if n_labels > 1:
            # Multilabel scenario as for FSC
            modified_trg = [logits_modified[i][:, targets[i]] for i in range(n_labels)]
            original_gt = [logits_original[i][:, targets[i]][0] for i in range(n_labels)]
            if single_perturbation_value is not None:
                # if we use single value of perturbation we don't need an average
                prediction_diff = [original_gt[i] - modified_trg[i] for i in range(n_labels)]
                print(f"Prediction diff: {prediction_diff}")
            else:
                prediction_diff = [original_gt[i] - np.mean(modified_trg[i]) for i in range(n_labels)]
        else:
            modified_trg = logits_modified[:, targets]
            original_gt = logits_original[:, targets][0]
            if single_perturbation_value is not None:
                # if we use single value of perturbation we don't need an average
                prediction_diff = [original_gt - modified_trg]
            else:
                # average prediction across all perturbed audios
                prediction_diff = [original_gt - np.mean(modified_trg)]

        if verbose:
            _tmp_log1(verbose_target, original_gt, modified_trg, n_labels)
            _tmp_log2(verbose_target, original_gt, modified_trg, n_labels)

        # Extract word/phoneme features
        if word_level:
            features = [word["word"] for word in words]
        elif phonemization:
            features = [word["char"] for word in words]
        else:
            features = [perturbation_type]

        # reshape to match perturbation values
        scores = np.array(prediction_diff)
        # print(f"Scores structure: {scores}")    # 2D array with scores for each phoneme and target (for FSC 3x16)
        # print(f"Features structure: {features}")        # 1D array with the phonemes/words

        target_classes_show = self.model_helper.get_text_labels_with_class(targets)

        # first visualization for phonemes/words
        if phonemization or word_level:
            # Bar chart for single perturbation value of all phoneme/words
            plot_phoneme_importance_bar(features, scores, target_classes_show)

        # to perturbate all phonemes/words
        if complete_perturbation:
            if word_level or phonemization:
                perturbed_audios_all, words_all, perturbations_all = self.perturb_waveform_words(
                    audio_path,
                    perturbation_type,
                    return_words=True,
                    verbose=verbose,
                    verbose_target=verbose_target,
                    phonemization=phonemization,
                    window_size=window_size,
                    respect_word_boundaries=respect_word_boundaries,
                )

                print(f"perturbed_audios_all: {len(perturbed_audios_all)}")

                # Get logits for modified audios, no need to recompute the original
                logits_modified_all = self.model_helper.predict(perturbed_audios_all)

                # Compute importance scores for top phonemes/words
                if n_labels > 1:
                    # initialize array to store result in correct form from the get-go
                    scores_all = np.zeros((n_labels, len(features), len(perturbations_all)))
                    # compute scores for each label
                    for label_idx in range(n_labels):
                        modified_trg = logits_modified_all[label_idx][:, targets[label_idx]]
                        print("modified_trg: ", modified_trg)
                        # reshape to (n_features, n_perturbations)
                        modified_trg = modified_trg.reshape(len(features), len(perturbations_all))
                        # calculate difference from original for each feature and perturbation
                        scores_all[label_idx] = original_gt[label_idx] - modified_trg
                    # ORIGINAL VERSION (doesn't work properly?)
                    # modified_trg_all = [logits_modified_all[i][:, targets[i]] for i in range(n_labels)]
                    # # print(f"modified_trg_all: {modified_trg_all}")      # debug print
                    # # modified_trg_all shape: list of n_labels (3) arrays, each array shape: (n_perturbed_audios,)
                    # # n_perturbed_audios = (perturbations*features)
                    # prediction_diff_all = [original_gt[i] - modified_trg_all[i] for i in range(n_labels)]
                    # # print(f"prediction_diff_all: {prediction_diff_all}")       # debug print
                    # # prediction_diff_all shape: list of n_labels arrays, each array shape: (n_perturbed_audios,)
                else:
                    scores_all = np.zeros((1, len(features), len(perturbations_all)))
                    modified_trg = logits_modified_all[:, targets]
                    modified_trg = modified_trg.reshape(len(features), len(perturbations_all))
                    scores_all[0] = original_gt - modified_trg
                    # modified_trg_all = logits_modified_all[:, targets]
                    # # modified_trg_all shape: (n_perturbed_audios,)
                    # prediction_diff_all = [original_gt - modified_trg_all]
                    # # prediction_diff_all shape: list with 1 array of shape (n_perturbed_audios,)

                # print(f"prediction_diff_top shape: {np.array(prediction_diff_all).shape}") # debug print
                # print(f"scores_all shape: {scores_all.shape}")      # debug print

                # Calculate the average effect across all perturbations
                avg_scores = np.mean(scores_all, axis=2)
                # print(f"avg_scores shape: {avg_scores.shape}")  # (n_labels, n_features)

                for label_idx, target in enumerate(target_classes_show):
                    print(f"Visualizing results for label {target}")
                    plot_perturbation_trends(features, scores_all[label_idx], perturbations_all, target)
                    plot_phoneme_importance_heatmap(features, scores_all[label_idx], perturbations_all, target)

                scores = avg_scores

                # print(f"scores dimensions:{scores.ndim}")

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=f"{self.NAME}+perturbation type:{perturbation_type}+window:{window_size}",
            target=targets if n_labels > 1 else [targets],
            audio_path=audio_path,
        )

        # debug prints
        # print(f"Structure of explanations: {explanation}")

        return explanation, perturbed_audios


    def explain_variations(self, audio_path, perturbation_types, target_class=None):
        """ Explains variations in the model's predictions by applying a different perturbation,
            this returns a dictionary of DataFrames where each of them contains the probability variations
            for each perturbation type. For a target class(es) we see how the prediction changes.
            Each dataframe has the perturbation values and the model predictions probabilities for each class under
            each perturbation type.
            """

        # Get logits for original audio for each class
        n_labels = self.model_helper.n_labels
        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]
        original_gt = self.model_helper.get_predicted_probs(audio=audio)

        if target_class is None:
            targets = self.model_helper.get_predicted_classes(audio=audio)
        else:
            targets = target_class

        target_classes_show = self.model_helper.get_text_labels(targets)
        print(f"target_classes_show: {target_classes_show}")

        perturbation_df_by_type = {}
        for perturbation_type in perturbation_types:
            # perturb the whole waveform
            perturbed_audios, perturbations = self.perturb_waveform(
                audio_path, perturbation_type, return_perturbations=True
            )

            if "time stretching" in perturbation_type:
                reference_value = 1
            else:
                reference_value = 0
                if "noise" == perturbation_type and USE_ADD_NOISE_TORCHAUDIO:
                    reference_value = 100

            prob_variations = []
            for perturbated_audio in perturbed_audios:
                probs_modified = self.model_helper.predict([perturbated_audio])

                if n_labels > 1:
                    # Multilabel scenario as for FSC
                    prob_variations.append(
                        [probs_modified[i][:, targets[i]][0] for i in range(n_labels)]
                    )
                else:
                    prob_variations.append([probs_modified[:, targets][0]])

            if n_labels > 1:
                prob_variations.append(original_gt)
            else:
                prob_variations.append([original_gt])

            x_labels = perturbations + [reference_value]
            prob_variations = np.array(prob_variations)[np.argsort(x_labels)]
            x_labels = np.array(x_labels)[np.argsort(x_labels)]

            if perturbation_type == "noise" and USE_ADD_NOISE_TORCHAUDIO:
                x_labels = x_labels[::-1]
                prob_variations = prob_variations[::-1]

            x_labels = [
                x_label if x_label != reference_value else REFERENCE_STR
                for x_label in x_labels
            ]

            # perturbations as columms, target class as rows
            perturbation_df = pd.DataFrame(prob_variations.T, columns=x_labels)
            perturbation_df.index = (
                target_classes_show if n_labels > 1 else [target_classes_show]
            )
            perturbation_df_by_type[perturbation_type] = perturbation_df
        return perturbation_df_by_type

    def _tmp_log_show_info(
        self,
        perturbation_type: str,
        perturbation_value: float,
        perturbated_audio: np.ndarray,
        verbose_target: int,
    ):
        from IPython.display import Audio

        # Display the perturbed audio and show its info for a single class
        # For multi label scenario, we show it for a single class: verbose_target
        # Note that in a single label scenario, verbose_target is ignored (always 0)

        print_log(perturbation_type, perturbation_value)
        # Prediction probability
        predictions = self.model_helper.predict([perturbated_audio])

        # Predicted label (idx)
        predicted_labels = [v.argmax() for v in predictions]

        # Predicted label (text)
        preds = self.model_helper.get_text_labels(predicted_labels)

        if self.model_helper.n_labels > 1:
            print_log(f"Target label: {verbose_target}")
            print_log(
                f"Predicted probs:",
                np.round(predictions[verbose_target], 3),
            )
            print_log(
                "Predicted class: ",
                preds[verbose_target],
                f"id: {predicted_labels[verbose_target]}",
            )
        else:
            print_log(
                f"Predicted probs: ",
                np.round(predictions[0], 3),
            )
            print_log(
                "Predicted class: ",
                preds,
                f"id: {predicted_labels[0]}",
            )
        display(Audio(np.transpose(perturbated_audio), rate=16000))
