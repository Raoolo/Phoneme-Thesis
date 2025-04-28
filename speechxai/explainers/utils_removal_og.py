from pydub import AudioSegment
from phonemizer import phonemize
import whisperx
import os
import numpy as np
from typing import Dict, List, Union, Tuple
from speechxai.explainers.visualization import (
    visualize_whisperx_results,
    plot_phoneme_importance_heatmap,
    plot_perturbation_trends,
    plot_phoneme_importance_bar, tmp_log_show_info)
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

from speechxai.utils import print_log, pydub_to_np


def remove_specified_words(audio, words, removal_type: str = "nothing", phonemization: bool = False):
    """
    Remove a word or phoneme from audio using pydub, by replacing it with:
    - nothing
    - silence
    - white noise
    - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        words: word or phoneme to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """
    from copy import deepcopy

    audio_removed = deepcopy(audio)

    if phonemization:
        a, b, = 5, 5
    else:
        a, b = 100, 40

    for word in words:
        start = int(word["start"] * 1000)
        end = int(word["end"] * 1000)

        before_word_audio = audio_removed[: start - a]
        after_word_audio = audio_removed[end + b :]

        word_duration = (end - start) + a + b

        if removal_type == "nothing":
            replace_word_audio = AudioSegment.empty()
        elif removal_type == "silence":
            replace_word_audio = AudioSegment.silent(duration=word_duration)
        elif removal_type == "white noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]
        elif removal_type == "pink noise":
            sound_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
            replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        audio_removed = before_word_audio + replace_word_audio + after_word_audio
    return audio_removed


def add_word_indices_to_chars(segments):
    """ Add word indices to chars to track which word each phoneme belongs to.
        Used for aggregation of phonemes. """

    chars_with_indices = []
    for segment in segments:
        words = segment.get('words')
        chars = segment.get('chars')

        for char in chars:
            # find which word this char belongs to
            char_mid_time = (char['start'] + char['end']) / 2
            word_idx = None

            for idx, word in enumerate(words):
                if word['start'] <= char_mid_time <= word['end']:
                    word_idx = idx
                    break

            char['word_idx'] = word_idx
            chars_with_indices.append(char)

    return chars_with_indices

def transcribe_audio(
    audio_path: str,
    device: str = "cuda",
    batch_size: int = 2,
    compute_type: str = "float32",
    language: str = "en",
    model_name_whisper: str = "large-v3-turbo",
    phonemization: bool = False,      # used to choose between words or phoneme level
    window_size: int = None,
    respect_word_boundaries: bool = True,
    visualization: bool = False,
    sliding: bool = False,
) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using whisperx,
    and return the text (transcription) and the words with their start and end times.
    """
    ## Load audio
    audio = whisperx.load_audio(audio_path)
    ## Load whisperx model and transcribe
    model_whisperx = whisperx.load_model(model_name_whisper, device=device, compute_type=compute_type, language=language)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)
    # print(f"Whisper segments og model: {result['segments']}")
    # print(f"Result structure: {result}")        # debug print, it's a dictionary with segments and language as keys
    # Result structure : {'segments': [{'text': ' Turn up the bedroom heat.', 'start': 0.436, 'end': 1.55}], 'language': 'en'}
    og_text = result["segments"][0]["text"]

    # transcription is improved with phonemizer to get the IPA phonemic transcription
    if phonemization:
        # Create phone transcript using both Wav2Vec2 and phonemizer results
        phone_transcript = [{"text": phonemize(segment["text"]), "start": segment["start"], "end": segment["end"]}
                            for segment in result["segments"]]
        # print(f"Phone_transcripts structure: {phone_transcript}")
        # Phone_transcripts structure: [{'text': 'tɜːn ʌp ðə bɛdɹuːm hiːt ', 'start': 0.436, 'end': 1.55}]

        # also try the alignment without phonemization
        # this is a model fine-tuned for phoneme alignment model_name="facebook/wav2vec2-lv-60-espeak-cv-ft"
        # other models to try: charsiu/en_w2v2_fs_20ms , charsiu/en_w2v2_fs_10ms, charsiu/en_w2v2_fc_10ms_32k,
        # charsiu/en_w2v2_fc_20ms, charsiu/en_w2v2_fc_10ms
        # from https://github.com/lingjzhu/charsiu
        # ct-vikramanantha/phoneme-scorer-v2-wav2vec2
        model_a, metadata = whisperx.load_align_model(model_name="facebook/wav2vec2-lv-60-espeak-cv-ft", language_code=result["language"], device=device)
        # if we use no model it might not even be able to detect some words

        # align timestamps, The return_char_alignments parameter enables phoneme level alignments
        result = whisperx.align(
            phone_transcript,  # phoneme
            # result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=phonemization,  # for phoneme alignment -> True
        )

        if visualization:
            visualize_whisperx_results(result["segments"], phonemization=phonemization, og_text=og_text)

        print("WhisperX result['segments'] structure before removal:", result["segments"])  # which contains start, end, text, words and chars dict

        # Remove chars with no timestamp such as ː that can come before or after a phoneme
        for segment in result["segments"]:
            segment["chars"] = [char for char in segment["chars"] if "start" in char]
            # this modifies the middle point of the phoneme making it out of bounds for the word index
            # if segment["chars"][-1]["end"] != phone_transcript[-1]["end"]:
            #     segment["chars"][-1]["end"] = phone_transcript[-1]["end"]  # make them the same lenght for the last phoneme

    else:   # word level
        model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
        result = whisperx.align(
            result['segments'],     # words
            model_a,
            metadata,
            audio,
            device,
        )

        if visualization and not sliding:
            visualize_whisperx_results(result["segments"], phonemization=phonemization, og_text=og_text)

        for segment in result["segments"]:
            segment["words"] = [char for char in segment["words"] if "start" in char]


    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    # Debug print
    # print("WhisperX result structure:", result)     # it's a dict that contains one segment equivalent to a sentence...
    # print("WhisperX result['segments'] structure:", result["segments"])   # which contains start, end, text, words and chars dict
    # and both words and chars dictionary have word/char, start, end and a score (confidence about that interval)

    if len(result["segments"]) == 1:        # if there is only one sentence = segment
        text = result["segments"][0]["text"]
        # words = result["segments"][0]["words"]
        if phonemization:
            if window_size:
                chars = add_word_indices_to_chars([result["segments"][0]])
                chars = aggregate_phonemes(chars, window_size, respect_word_boundaries, sliding=sliding)
            else:       # no phoneme aggregation
                chars = result["segments"][0]["chars"]
        else:
            chars = result["segments"][0]["words"]
    else:   # multiple sentences case
        text = " ".join(result["segments"][i]["text"] for i in range(len(result["segments"])))
        # words = [word for segment in result["segments"] for word in segment["words"]]
        if phonemization:
            if window_size:
                chars = add_word_indices_to_chars(result["segments"])
                chars = aggregate_phonemes(chars, window_size, respect_word_boundaries, sliding=sliding)
            else:
                chars = [char for segment in result["segments"] for char in segment["chars"]]
        else:
            chars = [word for segment in result["segments"] for word in segment["words"]]

    # # debug print
    # print(f"text: {text}")
    # print(f"chars: {chars}")      # list with dictionaries

    # text contains the sentence with timestamps, chars contain phoneme/words with timestamps
    return text, chars

def aggregate_phonemes(chars: List[Dict], window_size: int = 2, respect_word_boundaries: bool = True, sliding: bool = False) -> List[Dict]:
    """
    Aggregate phonemes using a sliding window approach.

    Args:
        chars: List of phoneme dictionaries with 'start', 'end', 'char', etc.
        window_size: Number of phonemes to aggregate together
        respect_word_boundaries: If True, won't aggregate phonemes across words
    """
    if window_size <= 1:        # no window aggregation
        return chars

    aggregated_chars = []       # save aggregated chars in this list of dictionaries
    words_aggregated = set()    # to avoid the aggregation < window_size at the end of the words
    # if not sliding:
    i = 0
    while i < len(chars):
        window = []     # initialize current window and control progression through chars
        current_word = None if not respect_word_boundaries else chars[i].get('word_idx')
        j = i   #build window from position i

        while len(window) < window_size and j < len(chars):     # check not to go over chars
            if respect_word_boundaries:
                if chars[j].get('word_idx') != current_word:    # check if still in same word
                    break   # and break second while in case
            window.append(chars[j])     # otherwise start building current window
            j += 1      # move position

        if window:  # if we have phonemes in our window
            if len(window) == window_size:
                aggregated_chars.append({
                    'char': ''.join(ph['char'] for ph in window),
                    'start': window[0]['start'],        # first phoneme
                    'end': window[-1]['end'],       # last phoneme of window
                    'score': sum(p.get('score', 0) for p in window) / len(window),      # average score
                    'word_idx': window[0].get('word_idx'),
                    # 'is_aggregated': True,        # debug
                })
                if respect_word_boundaries:
                    words_aggregated.add(current_word)
            else: # incomplete window, if at the end of the word ignore it
                if not sliding:
                    aggregated_chars.append({
                        'char': ''.join(ph['char'] for ph in window),
                        'start': window[0]['start'],
                        'end': window[-1]['end'],
                        'score': sum(p.get('score', 0) for p in window) / len(window),
                        'word_idx': current_word,
                    })
                else:
                    if not respect_word_boundaries or current_word not in words_aggregated:
                        aggregated_chars.append({
                            'char': ''.join(ph['char'] for ph in window),
                            'start': window[0]['start'],
                            'end': window[-1]['end'],
                            'score': sum(p.get('score', 0) for p in window) / len(window),
                            'word_idx': current_word,
                        })
                if respect_word_boundaries:
                    words_aggregated.add(current_word)

            # print(f"agg_char: {agg_char}")      # debug print
        if not sliding:
            i = j  # move to next window, if i want to have sliding window use i += 1
        else:
            i += 1  # this should give "la" "ai" "it" "ts" "s" instead of "la" "it" "s" fow windows_size = 2

    # print(f"aggregated_chars: {aggregated_chars}")      # debug print

    return aggregated_chars



def transcribe_audio_given_model(
    model_whisperx,
    audio_path: str,
    batch_size: int = 2,
    device: str = "cuda",
    phonemization: bool = False,

) -> Tuple[str, List[Dict[str, Union[str, float]]]]:
    """
    Transcribe audio using a specified whisperx model,
    and return the text (transcription) and the words with their start and end times.
    """

    ## Transcribe audio
    audio = whisperx.load_audio(audio_path)
    result = model_whisperx.transcribe(audio, batch_size=batch_size)

    if phonemization:
        phone_transcript = [{"text": phonemize(segment["text"]), "start": segment["start"], "end": segment["end"]}
                            for segment in result["segments"]]

        model_a, metadata = whisperx.load_align_model(
            model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft",
            language_code=result["language"],
            device=device
        )

        result = whisperx.align(
            phone_transcript,
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=True,
        )

        for segment in result["segments"]:
            segment["chars"] = [char for char in segment["chars"] if "start" in char]

    else:
        model_a, metadata = whisperx.load_align_model(
            language_code=result["language"],
            device=device
        )

        result = whisperx.align(
            result["segments"],
            model_a,
            metadata,
            audio,
            device,
            return_char_alignments=False,
        )

        for segment in result["segments"]:
            segment["words"] = [word for word in segment["words"] if "start" in word]

    if result is None or "segments" not in result or len(result["segments"]) == 0:
        return "", []

    if len(result["segments"]) == 1:
        text = result["segments"][0]["text"]
        chars = result["segments"][0]["chars"] if phonemization else result["segments"][0]["words"]
    else:
        text = " ".join(segment["text"] for segment in result["segments"])
        if phonemization:
            chars = [char for segment in result["segments"] for char in segment["chars"]]
        else:
            chars = [word for segment in result["segments"] for word in segment["words"]]

    return text, chars

def augmentation(
    perturbation_type: str,
    perturbation_value: float,
) -> Compose:
    """
    Creating a list of augmentations for each perturbation type
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


def remove_phoneme_word(audio_as,
                        word: dict = None,
                        perturbation_type: str = "nothing",
                        phonemization: bool = False,
                        perturbations: list = [],
                        verbose: bool = False,
                        verbose_target: int = 0,
                        model_helper = None,
                        paraling: bool = False,
    ):
    """
    Remove a word from audio using pydub, by replacing it with:
        - nothing
        - silence
        - white noise
        - pink noise

    Args:
        audio (pydub.AudioSegment): audio
        word: word to remove with its start and end times
        removal_type (str, optional): type of removal. Defaults to "nothing".
    """

    audio, frame_rate = pydub_to_np(audio_as)

    if verbose:
        # Display the original audio and show its info for a single class
        print_log("Original audio")
        tmp_log_show_info(
            "Original audio",
            "",
            audio.squeeze(),
            verbose_target=verbose_target,
            model_helper=model_helper,
        )

    if phonemization:
        a, b = 5, 5
    else:
        a, b = 100, 40

    #print(f"paraling: {paraling}")
    if paraling:
        # for perturbations
        start_ms = word["start"] * 1000  # convert to milliseconds
        end_ms = word["end"] * 1000
        perturbed_audios = []       # if len perturbations > 1

        # extract the word segment
        word_segment = audio_as[start_ms:end_ms]       # this works on audiosegment
        word_audio, _ = pydub_to_np(word_segment)
        # print(f"len before perturbation: {len(word_audio)}")
        for perturbation_value in perturbations:
            if "time stretching" in perturbation_type:
                # audiostretchy works on the full audio instead of segments so we don't use it
                # the following also doesn't work for phonemes, they are too short
                # Method 0 with audio_effects
                # perturbed_word = self.time_stretching_augmentation(word_segment, perturbation_value)
                # Method 1 with audiomentations (works but very low scores?, also seems to do opposite of value)
                augment = augmentation(perturbation_value=perturbation_value,
                                            perturbation_type=perturbation_type)
                perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)
                # Method 2 with rubberband (comparable results of M1)
                # perturbed_word = self.time_stretch_phoneme(word_audio, perturbation_value, sample_rate=frame_rate)
            elif "pitch shifting" in perturbation_type:
                # Method 1
                # perturbed_word = self.change_pitch_torchaudio(word_audio, frame_rate, perturbation_value)
                # Method 2 with audiomentations, MUCH faster and comparable results for multiple perturbations
                augment = augmentation(perturbation_value=perturbation_value,
                                            perturbation_type=perturbation_type)
                perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)
            else:
                augment = augmentation(perturbation_value=perturbation_value,
                                            perturbation_type=perturbation_type)
                perturbed_word = augment(samples=word_audio.squeeze(), sample_rate=frame_rate)

            # debug print
            #print(f"Perturbed_words shape: {perturbed_word.shape}")
            # Reshape to match the expected shape (n_samples, 1)
            perturbed_word = perturbed_word.reshape(-1, 1)
            #print(f"shape after perturbation: {len(perturbed_word)}")

            # Reconstruct the audio with the perturbed word/phoneme
            perturbed_audio = np.concatenate([
                audio[:int(start_ms * frame_rate / 1000)],  # audio before the phoneme
                perturbed_word,  # perturbed word/phoneme
                audio[int(end_ms * frame_rate / 1000):]  # audio after the phoneme
            ])

            # print(f"perturbed_audio paraling: {type(perturbed_audio)}")
            # MBY DEVO AGGIUNGERE NORMALIZZAZIONE DI PERTURBED AUDIO (no spe dovrebbe essere gia' a  posto?)
            # e sistemare il time_stretching con i tempi yk


            if verbose:
                # Display the perturbed audio and show its info for a single class
                tmp_log_show_info(
                    perturbation_type,
                    perturbation_value,
                    perturbed_audio,
                    verbose_target,
                )

            if len(perturbations) > 1:
                perturbed_audios.append(perturbed_audio)

    else:
        # split for removal
        before_word_audio = audio_as[: word["start"] * 1000 - a]
        after_word_audio = audio_as[word["end"] * 1000 + b :]
        word_duration = (word["end"] * 1000 - word["start"] * 1000) + a + b

        if perturbation_type == "nothing":
            replace_word_audio = AudioSegment.empty()
        elif perturbation_type == "silence":
            replace_word_audio = AudioSegment.silent(duration=word_duration)
        # elif perturbation_type == "white noise":
        #     sound_path = (os.path.join(os.path.dirname(__file__), "white_noise.mp3"),)
        #     replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]
        #     # display(audio_removed)
        # elif perturbation_type == "pink noise":
        #     sound_path = (os.path.join(os.path.dirname(__file__), "pink_noise.mp3"),)
        #     replace_word_audio = AudioSegment.from_mp3(sound_path)[:word_duration]

        perturbed_audio = before_word_audio + replace_word_audio + after_word_audio     # this returns an audiosegment
        #print(f"perturbed_audio before: {type(perturbed_audio)}")
        perturbed_audio = pydub_to_np(perturbed_audio)[0]   # idk if this works for perturbations
        print(f"perturbed_audio after: {type(perturbed_audio)}")

    if len(perturbations) > 1:
        return perturbed_audios
    else:
        return perturbed_audio
