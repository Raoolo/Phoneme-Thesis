import numpy as np
import shap
import matplotlib.pyplot as plt
import os
from pydub import AudioSegment
from copy import deepcopy
from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.visualization import plot_lime_heatmap
from speechxai.utils import pydub_to_np
from speechxai.explainers.utils_removal import transcribe_audio, remove_phoneme_word

class SHAPSpeechExplainer:
    NAME = "SHAP"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
            self,
            audio_path: str,
            target_class=None,
            words_transcript: list = None,
            removal_type: str = "silence",
            num_samples: int = 100,
            aggregation: str = "sum",   # to remove
            phonemization: bool = False,
            window_size: int = None,
            respect_word_boundaries: bool = True,
            visualization: bool = False,
            sliding: bool = False,
    ) -> ExplanationSpeech:

        """
        Compute a word/phoneme-level SHAP explanation
        """

        if removal_type not in ["silence", "nothing"]:
            raise ValueError("Removal method not supported, choose between 'silence' and 'nothing'")

        # words_transcript can contain either words or phonemes
        if words_transcript is None:
            _, words_transcript = transcribe_audio(
                audio_path=audio_path,
                language=self.model_helper.language,
                phonemization=phonemization,
                window_size=window_size,
                respect_word_boundaries=respect_word_boundaries,
                sliding=sliding
            )

        # load the audio as numpy array
        audio_segment = AudioSegment.from_wav(audio_path)
        audio_numpy = pydub_to_np(audio_segment)[0]
        # print("Audio shape:", audio_numpy.shape)  # debug print

        # Predict logits/probabilities for original audio
        logits_original = self.model_helper.predict([audio_numpy])

        # Check if single label or multilabel scenario as for FSC
        n_labels = self.model_helper.n_labels
        if target_class is not None:
            targets = target_class
        else:
            if n_labels > 1:
                # Multilabel scenario as for FSC
                targets = [int(np.argmax(logits_original[i], axis=1)[0]) for i in range(n_labels)]
            else:
                targets = [int(np.argmax(logits_original, axis=1)[0])]

        # working on features instead of frames
        features = [item["char"] if phonemization else item["word"] for item in words_transcript]
        # print(f"features: {features}")      # debug print, ['tɜn', 'ɔf', 'ðə', 'laɪ', 'ts']
        n_features = len(features)

        # this function masks words/phonemes then return predictions
        # this function is passed to SHAP method
        def prediction_for_masks(binary_mask):
            """
            Prediction function that takes a binary mask of features and returns model predictions.
            Each feature is represented by a number, 1 = present and 0 = absent.
            """

            # print(f"binary mask: {binary_mask}")        # debug print, [[0 1 0 0 0 1], [1 0 0 1 0 1], ...]]
            # print(f"binary_mask.shape = {binary_mask.shape}")   # debug print,
            # (1000, 6) -> binary mask, (1, 6) -> instance to explain, (62000, 6) -> ? idk probably due to l1_reg

            if len(binary_mask.shape) == 1:     # in case single mask
                return self._process_mask(          # process the mask (remove features) and makes predictions
                    mask=binary_mask, audio_path=audio_path, words_transcript=words_transcript,
                    removal_type=removal_type, phonemization=phonemization, targets=targets
                )
            else:       # multiple masks
                results = []
                for mask in binary_mask:
                    pred = self._process_mask(
                        mask=mask, audio_path=audio_path, words_transcript=words_transcript,
                        removal_type=removal_type, phonemization=phonemization, targets=targets,
                    )
                    results.append(pred)
                return np.array(results)

        # default number of samples in case not given
        if num_samples is None or num_samples <= 0:
            num_samples = n_features * 10

        # the missing features during importance calculations are set to the background value, thus this
        # mimics what an input looks like when some of the features are missing
        # = samples with different combinations of features (binary vector where 1 means the feature is present)
        background_data = np.random.randint(0, 2, size=(num_samples, n_features))     # 50% of being included
        # print(f"Background data: {background_data}")    # debug print, [[0 1 0 0 0 1], [0 0 0 1 0 0 1], ... ]]
        # print("background_data shape", background_data.shape)       # debug print, (1000, 6)

        # https://shap.readthedocs.io/en/latest/generated/shap.KernelExplainer.html
        # uses a weighted linear regression to compute the importance of each feature (shapley values)
        # requires the prediction model that we want to explain
        # background to substitute missing features
        # link can be "identity" or "logits" (for logits or probabilities idk they say different things)

        explainer = shap.SamplingExplainer(model=prediction_for_masks, data=background_data)

        # current instance=all features present
        instance = np.ones((1, n_features))     # [1, 1, 1, 1, 1, 1]

        # compute SHAP values for the audio converted
        # if l1_reg="num_features(int)" is used it should be tuned for any different run, for phonemization it could be 5
        # for windows it could be 2/3 and for words maybe 1-2, otherwise just leave it as is
        shap_values = explainer.shap_values(instance)
        # print(f"Shap values len: {len(shap_values)}")  # debug prints, 3 for fsc
        # print(f"Shap values: {shap_values}")  # debug prints, each object is size (num_samples, n_features)
        # print(f"Shap values shape: {shap_values[0].shape}")  # debug prints, (1, 6)

        # reshape shap_values
        if n_labels > 1:
            # for multi-label shap_values is a list of arrays (one per label)
            shap_array = np.array([values[0] for values in shap_values])
        else:
            # for single label we needt to wrap it in an extra dimension for later use
            shap_array = np.array([shap_values[0]])
        # print(f"Shap array shape after reshape: {shap_array.shape}")

        explanation = ExplanationSpeech(
            features=features,
            scores=shap_array,
            explainer=f"{self.NAME}+removal_type:{removal_type}+window:{window_size}",
            target=targets,
            audio_path=audio_path,
        )

        if visualization:
            # built-in SHAP visualization
            if n_labels == 1:
                plt.figure(figsize=(10, 6))
                shap.bar_plot(shap_array[0], features=features, show=False)
                plt.title(f"SHAP values for {self.model_helper.get_text_labels_with_class(targets)[0]}")
                plt.tight_layout()
                plt.savefig(os.path.join("./plots/", "shap_results"))
                plt.show()
            else:
                # prova con custom
                target_labels = self.model_helper.get_text_labels_with_class(targets)
                for i, label in enumerate(target_labels):
                    plt.figure(figsize=(10, 6))
                    shap.bar_plot(shap_array[i], features=features, show=False)
                    plt.title(f"SHAP values for {label}")
                    plt.tight_layout()
                    plt.savefig(os.path.join("./plots/", "shap_results"))
                    plt.show()
                # fig = plot_lime_heatmap(explanation, target_labels)
                # plt.savefig(os.path.join("./plots/", "shap_results"))
                # plt.show()

        return explanation

    def _process_mask(
            self,
            mask,
            audio_path: str,
            words_transcript: list = None,
            removal_type: str = "silence",
            phonemization: bool = False,
            targets: list = None,
    ):
        # convert the mask to a list of indices to remove, take the 0s
        indices_to_remove = [i for i, val in enumerate(mask) if val == 0]       # if val is 0, then add the idx to list
        # print(f"indices_to_remove: {indices_to_remove}")      # debug print
        if not indices_to_remove:
            # keeping all features = use the original audio
            audio_data = pydub_to_np(AudioSegment.from_wav(audio_path))[0]
            logits = self.model_helper.predict([audio_data])
        else:
            # remove some features from the audio
            modified_audio = self._remove_features(
                audio_path, words_transcript, indices_to_remove,
                removal_type, phonemization
            )
            logits = self.model_helper.predict([modified_audio])

        # extract probabilities for target classes
        n_labels = self.model_helper.n_labels
        if n_labels > 1:
            # for multi-label extract probability for each target
            return np.array([logits[i][0, targets[i]] for i in range(n_labels)])
        else:
            return np.array([logits[0, targets[0]]])

    def _remove_features(
            self,
            audio_path: str,
            words_transcript: list = None,
            indices_to_remove: list = None,
            removal_type: str = "silence",
            phonemization: bool = False,
    ):
        # this method is similar to remove_phoneme_word but that didn't work for this so i implemented a parallel here

        # load the audio file
        audio_segment = AudioSegment.from_wav(audio_path)

        # sort indices in reverse order because following positions would mess up when removing
        for idx in sorted(indices_to_remove, reverse=True):     # [1, 3, 5] -> [5, 3, 1,]
            feature = words_transcript[idx]     # take the idx-th feature

            if phonemization:
                buffer_start, buffer_end = 5, 5
            else:
                buffer_start, buffer_end = 100, 40

            # convert to ms
            start_ms = int(feature["start"] * 1000) - buffer_start
            end_ms = int(feature["end"] * 1000) + buffer_end

            # make start and end always valid, had problems with this don't know how utils_removal works without it
            start_ms = max(0, start_ms)
            end_ms = min(len(audio_segment), end_ms)

            # split for removal
            before_segment = audio_segment[:start_ms]
            after_segment = audio_segment[end_ms:]

            # replace feature
            segment_duration = end_ms - start_ms
            if removal_type == "nothing":
                replacement = AudioSegment.empty()
            elif removal_type == "silence":
                replacement = AudioSegment.silent(duration=segment_duration)

            audio_segment = before_segment + replacement + after_segment

        # convert to numpy array for prediction
        audio_numpy = pydub_to_np(audio_segment)[0]
        return audio_numpy