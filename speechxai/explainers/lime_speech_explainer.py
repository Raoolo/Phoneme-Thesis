from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.visualization import plot_lime_heatmap
from speechxai.utils import pydub_to_np
from typing import List
from pydub import AudioSegment
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from speechxai.explainers.lime_timeseries import LimeTimeSeriesExplainer

from speechxai.explainers.utils_removal import transcribe_audio

EMPTY_SPAN = "---"


class LIMESpeechExplainer:
    NAME = "LIME"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        words_transcript: List = None,
        removal_type: str = "silence",
        num_samples: int = 1000,
        phonemization: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        visualization: bool = False,
        sliding: bool = False,
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        """

        if removal_type not in ["silence", "noise", "mean", "total_mean"]:
            raise ValueError(
                "Removal method not supported, choose between 'silence', 'noise', 'mean' and 'total_mean'."
            )

        # Load audio and convert to np.array
        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        # Predict logits/probabilities for original audio
        logits_original = self.model_helper.predict([audio])

        # Check if single label or multilabel scenario as for FSC
        n_labels = self.model_helper.n_labels

        if target_class is not None:
            targets = target_class
        else:
            if n_labels > 1:
                # Multilabel scenario as for FSC
                targets = [
                    int(np.argmax(logits_original[i], axis=1)[0])
                    for i in range(n_labels)
                ]
            else:
                targets = [int(np.argmax(logits_original, axis=1)[0])]

        # words_transcript can contain either words or phonemes
        if words_transcript is None:
            # Transcribe audio
            _, words_transcript = transcribe_audio(
                audio_path=audio_path, language=self.model_helper.language,
                phonemization=phonemization, window_size=window_size, respect_word_boundaries=respect_word_boundaries,
                sliding=sliding,)
        audio_np = audio.reshape(1, -1)


        # Get the start and end indexes of the words. These will be used to split the audio and derive LIME interpretable features
        tot_len = audio.shape[0]
        sampling_rate = self.model_helper.feature_extractor.sampling_rate
        splits = []
        old_start = 0
        a, b = 0, 0
        if phonemization:
            for phoneme in words_transcript:
                # compute start and end time of phonemes
                start, end = int((phoneme["start"] + a) * sampling_rate), int( (phoneme["end"] + b) * sampling_rate)
                # add a segment for the gap between phonemes and one for the current phoneme
                # splits.append({"start": old_start, "end": start, "phoneme": EMPTY_SPAN})
                splits.append({"start": start, "end": end, "phoneme": phoneme["char"]})
                old_start = end     # starting point for next phoneme

            # final append from the last phoneme to the end of the audio
            # splits.append({"start": old_start, "end": tot_len, "phoneme": EMPTY_SPAN})
            # print(f"Splits structure: {splits}")        # debug print
            lime_explainer = LimeTimeSeriesExplainer(kernel_width=5)
        else:
            for word in words_transcript:
                start, end = int((word["start"] + a) * sampling_rate), int(
                    (word["end"] + b) * sampling_rate
                )
                splits.append({"start": old_start, "end": start, "word": EMPTY_SPAN})
                splits.append({"start": start, "end": end, "word": word["word"]})
                old_start = end

            splits.append({"start": old_start, "end": tot_len, "word": EMPTY_SPAN})
            lime_explainer = LimeTimeSeriesExplainer()

        # Compute gradient importance for each target label
        # This also handles the multilabel scenario as for FSC
        scores = []
        for target_label, target_class in enumerate(targets):
            if self.model_helper.n_labels > 1:
                # We get the prediction probability for the given label
                predict_proba_function = (
                    self.model_helper.get_prediction_function_by_label(target_label)
                )
            else:
                predict_proba_function = self.model_helper.predict

            input_audio = deepcopy(audio_np)

            # Explain the instance using the splits as interpretable features
            # print("len(splits): ", len(splits)) # debug print
            # print(f"Splits: {splits}")
            exp = lime_explainer.explain_instance(
                input_audio,
                predict_proba_function,
                num_features=len(splits),
                num_samples=num_samples,
                num_slices=len(splits),
                replacement_method=removal_type,
                splits=splits,
                labels=(target_class,),
            )

            # Extract the LIME explanation scores for the target class, k = feature index, v = importance score
            map_scores = {k: v for k, v in exp.as_map()[target_class]}
            # And sort the scores by feature index in ascending order to maintain temporal order
            map_scores = {k: v for k, v in sorted(map_scores.items(), key=lambda x: x[0], reverse=False)}

            # Remove the 'empty' segments between words/phonemes and map scores to phonemes
            if phonemization:       # map_scores is a list of tuples  [(phoneme1, score1), (phoneme2, score2), ...]
                map_scores = [
                    (splits[k]["phoneme"], v) for k, v in map_scores.items() if splits[k]["phoneme"] != EMPTY_SPAN
                ]   # map feature index (k) to phoneme and its score (v), excluding empty spans
                # print(f"target_label and class: {target_label} & {target_class}, map_scores structure: {map_scores}")    # debug print
            else:
                map_scores = [
                    (splits[k]["word"], v) for k, v in map_scores.items() if splits[k]["word"] != EMPTY_SPAN
                ]   # map feature index (k) to words and its score (v), excluding empty spans
                # print(f"target_label and class: {target_label} & {target_class}, map_scores structure: {map_scores}")      # debug print


            # if there are no phonemes with valid importance scores, set features and importances to empty
            if map_scores == []:
                features = []
                importances = []
            else:
                # if not empty, separate the phonemes and their importance scores
                # zip(*map_scores) transposes the list of tuples into two separate lists:
                features = list(list(zip(*map_scores))[0])      # contains the words/phonemes
                importances = list(list(zip(*map_scores))[1])   # contains importance scores
            scores.append(np.array(importances))

        if n_labels > 1:
            # Multilabel scenario as for FSC
            scores = np.array(scores)
        else:
            scores = np.array([importances])

        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=f"{self.NAME}+removal type:{removal_type}+window:{window_size}",
            target=targets if n_labels > 1 else targets,
            audio_path=audio_path,
        )

        if visualization:
            targets_labels = self.model_helper.get_text_labels_with_class(targets)
            fig = plot_lime_heatmap(explanation, targets_labels)
            plt.show()

        return explanation
