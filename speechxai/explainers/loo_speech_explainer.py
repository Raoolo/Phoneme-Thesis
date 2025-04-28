"""LOO Speech Explainer module"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Union, Tuple
from pydub import AudioSegment
from speechxai.explainers.visualization import plot_loo_importance, plot_phoneme_importance_bar, \
    plot_perturbation_trends, plot_phoneme_importance_heatmap
from speechxai.utils import pydub_to_np, print_log
from IPython.display import display
from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.utils_removal import transcribe_audio, remove_phoneme_word, aggregate_phonemes
from speechxai.explainers.paraling_speech_explainer import _tmp_log1, _tmp_log2


class LOOSpeechExplainer:
    NAME = "loo_speech"

    def __init__(self, model_helper):
        self.model_helper = model_helper

    def remove_phonemes_words(
            self,
            audio_path: str,
            removal_type: str = "nothing",
            chars: List = None,
            display_audio: bool = False,
            phonemization: bool = False,
            perturbation_list: List = None,
            verbose: bool = False,
            verbose_target: int = 0,
            model_helper=None,
            paraling: bool = False,
            single_perturbation_value: float = None,
    ) -> Tuple[List[AudioSegment], List[Dict[str, Union[str, float]]], np.ndarray]:
        """
        Remove words from audio using pydub, by replacing them with:
        ["nothing", "silence", "pitch shifting", "pitch shifting down", "pitch shifting up",
        "time stretching", "time stretching down", "time stretching up", "stress",
        "degradation", "noise", "white noise", "pink noise", "claps", "intensity", "reverberation"]
        """

        ## Load audio as pydub.AudioSegment
        audio_as = AudioSegment.from_wav(audio_path)
        ## List to store the audio with phonemes removed
        audio_no_phonemes = []

        perturbations = []
        if paraling:
            if single_perturbation_value is not None:
                perturbations = [single_perturbation_value]
            elif perturbation_list is not None:
                perturbations = perturbation_list
            else:
                if removal_type == "pitch shifting" or removal_type == "intensity":
                    perturbations = np.arange(-10, 11, 1)  # from doc [-12 to 12]
                elif removal_type == "pitch shifting down":
                    perturbations = np.arange(-10, 0.5, 0.5)
                elif removal_type == "pitch shifting up":
                    perturbations = np.arange(0.5, 10.5, 0.5)
                elif removal_type == "time stretching":  # from 0 up
                    perturbations = np.arange(0.25, 2.25, 0.25)
                elif removal_type == "time stretching down":
                    perturbations = np.arange(0.2, 1.1, 0.1)
                elif removal_type == "time stretching up":
                    perturbations = np.arange(1, 2.6, 0.1)
                elif removal_type == "reverberation":
                    # perturbation is equivalent to the meters of a room (in this case it will be a cube)
                    perturbations = np.arange(1, 6, 1)
                elif "noise" in removal_type or "claps" in removal_type:
                    perturbations = np.arange(-30, 30, 5)
                elif removal_type == "degradation":
                    perturbations = np.arange(1, 6, 1)

            print(f"perturbations: {perturbations}")


        # iterate through each word/phoneme in the list
        for char in chars:
            # for each phoneme, remove it and append the modified audio
            # debug prints
            # if phonemization:
                # print(f"Removing phoneme: {char['char']}, Start: {char['start']}, End: {char['end']}")
            # else:
            #     print(f"Removing word: {char['word']}, Start: {char['start']}, End: {char['end']}")

            audio_removed = remove_phoneme_word(
                                                audio_as, char, removal_type, phonemization,
                                                perturbations=perturbations, paraling=paraling,
                                                verbose=verbose, model_helper=model_helper,
                                                verbose_target=verbose_target,)
            if verbose:
                from IPython.display import Audio
                print_log("Original audio")
                # Display the original audio and show its info for a single class
                self._tmp_log_show_info(
                    "Original audio",
                    "",
                    audio_removed.squeeze(),
                    verbose_target,
                )

            if len(perturbations) > 1:
                audio_no_phonemes.extend(audio_removed)     # in this case audio removed is a list
            else:
                audio_no_phonemes.append(audio_removed)

            if display_audio:
                print_log(char["char"] if phonemization else char["word"])
                display(audio_removed)
        # print(f'audio_no_phonemes: {audio_no_phonemes}')

        return audio_no_phonemes, chars, perturbations


    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        removal_type: str = None,
        words_transcript: List = None,
        phonemization: bool = False,
        display_audio: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        single_perturbation_value: float = None,
        complete_perturbation: bool = False,
        visualization: bool = False,
        sliding: bool = False,
        perturbation_list: List = None,
        verbose: bool = False,
        verbose_target: int = 0,
    ) -> ExplanationSpeech:
        """
        Computes the importance of each word in the audio.
        """
        allowed_values = {
            "pitch shifting", "pitch shifting down", "pitch shifting up", "reverberation",
            "time stretching", "time stretching down", "time stretching up", "stress",
            "degradation", "noise", "white noise", "pink noise", "claps", "intensity",
            "nothing", "silence"}

        paraling_modifications = ["pitch shifting", "pitch shifting down", "pitch shifting up", "reverberation",
                                  "time stretching", "time stretching down", "time stretching up", "stress",
                                  "degradation", "noise", "white noise", "pink noise", "claps", "intensity", ]

        assert removal_type in allowed_values, f"Invalid values found: {removal_type}, values allowed: {allowed_values}"

        if removal_type in paraling_modifications and single_perturbation_value is None and not complete_perturbation:
            raise ValueError(
                "Cannot pick a perturbation type for a removal operation, choose between 'silence' and 'nothing'")

        paraling = False
        if removal_type in paraling_modifications:
            paraling = True

        if words_transcript is None:
            text, words_transcript = transcribe_audio(
                audio_path=audio_path,
                device=self.model_helper.device.type,
                batch_size=2,
                compute_type="float32",
                language=self.model_helper.language,
                phonemization=phonemization,
                window_size=window_size,
                respect_word_boundaries=respect_word_boundaries,
                visualization=visualization,
                sliding=sliding,
            )
        if text is False or words_transcript is False:
            return None
        # chars is a dictionary of phonemes/words, with keys char, start, end

        # predict logits for original audio
        audio, frame_rate = pydub_to_np(AudioSegment.from_wav(audio_path))
        logits_original = self.model_helper.predict([audio])

        # Get modified audios with a phoneme left out
        audio_no_phonemes, chars, perturbations = self.remove_phonemes_words(
            audio_path=audio_path,
            chars=words_transcript,
            removal_type=removal_type,
            display_audio=display_audio,
            phonemization=phonemization,
            perturbation_list=perturbation_list,
            paraling=paraling,
            model_helper=self.model_helper,
            verbose=verbose,
            verbose_target=verbose_target,
            single_perturbation_value=single_perturbation_value,
        )

        # predict logits for modified audios
        logits_modified = self.model_helper.predict(audio_no_phonemes)

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

        ## Get the most important word for each class (action, object, location), compute difference in prediction
        if n_labels > 1:
            # Multilabel scenario as for FSC
            modified_trg = [logits_modified[i][:, targets[i]] for i in range(n_labels)]
            original_gt = [logits_original[i][:, targets[i]][0] for i in range(n_labels)]
            if paraling:
                if single_perturbation_value is not None:
                    # if we use single value of perturbation we don't need an average
                    prediction_diff = [original_gt[i] - modified_trg[i] for i in range(n_labels)]
                    # print(f"Prediction diff: {prediction_diff}")
                else:
                    # average prediction across all perturbed audios (might need fixing)
                    prediction_diff = [original_gt[i] - np.mean(modified_trg[i]) for i in range(n_labels)]
            else:
                # Multilabel scenario as for FSC
                prediction_diff = [original_gt[i] - modified_trg[i] for i in range(n_labels)]
        else:
            modified_trg = logits_modified[:, targets]
            original_gt = logits_original[:, targets][0]
            if paraling:
                if single_perturbation_value is not None:
                    # if we use single value of perturbation we don't need an average
                    prediction_diff = [original_gt - modified_trg]
                else:
                    # average prediction across all perturbed audios
                    prediction_diff = [original_gt - np.mean(modified_trg)]
            else:
                prediction_diff = [original_gt - modified_trg]

        if verbose:
            _tmp_log1(verbose_target, original_gt, modified_trg, n_labels)
            _tmp_log2(verbose_target, original_gt, modified_trg, n_labels)

        # extract phonemes from the transcript and use them as features
        features = [char["char"] if phonemization else char["word"] for char in chars]
        # print(f"Features structure: {features}")        # 1D array with the phonemes/words

        scores = np.array(prediction_diff)
        # print(f"Scores structure: {scores}")    # 2D array with scores for each phoneme and target (for FSC 3x16)

        target_classes_show = self.model_helper.get_text_labels_with_class(targets)     # for the visualization

        if paraling:
            # Bar chart for single perturbation value of all phoneme/words
            plot_phoneme_importance_bar(features, scores, target_classes_show)


        # this is done to get heatmaps of how EACH perturbation level affects the outcome, while the previous code was
        # checking the average effect of the list of the perturbation
        if complete_perturbation and paraling:
            perturbed_audios_all, chars_all, perturbations_all = self.remove_phonemes_words(
                audio_path=audio_path,
                chars=words_transcript,
                removal_type=removal_type,
                display_audio=display_audio,
                phonemization=phonemization,
                perturbation_list=perturbation_list,
                paraling=paraling,
                model_helper=self.model_helper,
                verbose=verbose,
                verbose_target=verbose_target,
            )

            # print(f"perturbed_audios_all: {len(perturbed_audios_all)}")

            # Get logits for modified audios, no need to recompute the original
            logits_modified_all = self.model_helper.predict(perturbed_audios_all)

            # Compute importance scores for top phonemes/words
            if n_labels > 1:
                # initialize array to store result in correct form from the get-go
                scores_all = np.zeros((n_labels, len(features), len(perturbations_all)))
                # compute scores for each label
                for label_idx in range(n_labels):
                    modified_trg = logits_modified_all[label_idx][:, targets[label_idx]]
                    # print("modified_trg: ", modified_trg)
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
            #print(f"scores dimensions:{scores.ndim}")


        if paraling:
            explainer = f"{self.NAME}+perturbation_type:{removal_type}+window:{window_size}"
        else:
            explainer = f"{self.NAME}+removal_type:{removal_type}+window:{window_size}"


        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=explainer,
            target=targets if n_labels > 1 else [targets],
            audio_path=audio_path,
        )

        labels_names = self.model_helper.get_text_labels_with_class(targets)
        if visualization and not paraling:
            fig = plot_loo_importance(
                audio=audio,
                sampling_rate=self.model_helper.feature_extractor.sampling_rate,
                segments=chars,
                scores=scores.squeeze(),
                phonemization=phonemization,
                targets=labels_names
            )
            plt.show()

        # print(explanation)        # debug print

        return explanation


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