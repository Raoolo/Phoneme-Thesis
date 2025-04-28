import numpy as np
import matplotlib.pyplot as plt
from speechxai.explainers.visualization import plot_aopc_curve_compr, plot_aopc_curve_suff
from speechxai.utils import pydub_to_np
from pydub import AudioSegment
import warnings
from ferret.evaluators.utils_from_soft_to_discrete import (
    parse_evaluator_args,
    _check_and_define_get_id_discrete_rationale_function,
)
from speechxai.explainers.explanation_speech import ExplanationSpeech, EvaluationSpeech
from speechxai.explainers.utils_removal import remove_specified_words, transcribe_audio
from ferret.evaluators.faithfulness_measures import _compute_aopc
from typing import List

# faithfulness measures how well the explanations reflect the speech model. Both of the following methods
# use the AOPC (Average over Perturbation Curve).
# AOPC is a metric that averages the change in model's prediction over different levels of perturbation,
# meaning removing or retaining features at different thresholds (different levels of feature importance to determine
# which words/phoneme to remove or retain).

#compare the metrics of different explainers and determines which one has the most faithful explanations

class AOPC_Comprehensiveness_Evaluation_Speech:

    NAME = "aopc_comprehensiveness"
    SHORT_NAME = "aopc_compr"
    # Higher is better
    BEST_SORTING_ASCENDING = False
    TYPE_METRIC = "faithfulness"

    def __init__(self, model_helper, **kwargs):
        self.model_helper = model_helper

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target=None,
        words_transcript: List = None,
        phonemization: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        sliding: bool = False,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Comprehensiveness metric. Measures how much the prediction changes
        when the most important features are removed (the higher, the more important).
        Extracts the audios and the transcription, computes the model's prediction then removes the most important
        features (words/phoneme) from the audio and compute the prediction again. Computes the difference between
        and averages it across multiple thresholds to compute the AOPC comprehensiveness score.

        Args:
            explanation (Explanation): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation
        Returns:
            Evaluation : the AOPC Comprehensiveness score of the explanation
        """

        # from ferret documentation:
        # only_pos ->  As a default, we consider in the rationale only the terms influencing positively the prediction
        # removal_args -> As a default, we remove from 10% to 100% of the tokens.
        # structure: removal_args = {"remove_tokens": True,"based_on": "perc", "thresholds": np.arange(0.1, 1.1, 0.1)}
        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert ("perturb_paraling" not in explanation.explainer), f"{explanation.explainer} not supported"
        # print(f"explanation.explainer {explanation.explainer}")  # debug print

        if "+" in explanation.explainer:
            # The explainer name contain "+" to specify the removal type --> loo_speech+silence
            partial = explanation.explainer.split("+")[1]
            removal_type = partial.split(":")[1]
            # print(removal_type)
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        audio_path = explanation.audio_path
        target = explanation.target
        audio = AudioSegment.from_wav(audio_path)   # get the audio from audio_path
        audio_np = pydub_to_np(audio)[0]
        labels = self.model_helper.get_text_labels_with_class(target)

        # Get prediction probability of the input sentence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])
        # print("ground_truth_probs", ground_truth_probs)     # debug print
        # ground_truth_probs (array([[9.25510613e-10, 1.89430551e-07, 1.56134888e-07, 9.99869227e-01,
        # 1.30391098e-04, 8.84956641e-09]]), array([[1.41779232e-12, 8.12982928e-12, 3.92900716e-07, 2.87418607e-05,
        # 9.99970675e-01, 1.92245300e-11, 2.69004175e-07, 5.77682790e-10, 6.08114991e-12, 1.92484900e-11, 2.22094449e-08, 5.43362707e-12,
        # 1.35985334e-10, 6.84768377e-13]]), array([[1.63515005e-03, 5.10979007e-05, 9.97955680e-01, 3.58074176e-04]]))

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # print(f"ground_truth_probs_target: {ground_truth_probs_target}")        # debug print
        # ground_truth_probs_target: [0.9998692274093628, 0.9999706745147705, 0.9979556798934937]

        # Split the audio into word-level/phoneme-level audio segments
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
                sliding=sliding,
            )

        # print(f"len words_transcript: {len(words_transcript)}")

        # discrete rationale --> subset of features that are most important based on the explanation score
        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]        # based on = perc
            )
        )

        # "thresholds": np.arange(0.1, 1.1, 0.1) from the ferret documentation
        thresholds = removal_args["thresholds"]
        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level
        # or phoneme level explanations
        # or 1 in the case of paralinguistic audio level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation
        # print("score_explanations: ", score_explanations)
        # score_explanations:  [[0.34367895 0.22449489 0.3593378  0.6784782  0.8538744  0.17034635
        #   1.6527896  1.4775021  0.2003593  0.54604363 0.36717033 0.56480443
        #   0.5244094  0.18765727 0.30558315 2.2311673 ]
        #  [0.41089773 0.42064685 0.69537    0.899253   1.3516613  0.31251425
        #   2.1858501  2.5494413  0.4076993  1.1006801  0.63760823 1.5279737
        #   1.4699554  0.7930793  0.8575276  2.168511  ]
        #  [0.46315846 0.21265785 0.49409765 0.78953034 0.6146893  0.17657903
        #   1.0855485  2.5159912  0.34199783 0.8116315  0.39372948 0.652283
        #   0.70598906 0.3194802  0.39571747 1.5337318 ]]

        aopc_comprehensiveness_multi_label = list()
        scores_by_threshold = {t: [] for t in thresholds}
        removed_features_by_label = {}

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()        # store changes in prediction probability for thresholds
            id_tops = list()    # indices of most important features
            last_id_top = None      # last set of important features
            used_thresholds = list()  # track which thresholds were actually used
            removed_features = []  # track features removed at each threshold

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]     # model prediction for target class

            # We compute the difference in probability for all the thresholds
            for v in thresholds:
                # Get rationale (most important features) from score explanation for the current thresholds
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)
                # v is converted to an int and words are removed based on its value
                # print("Indices to remove: ", id_top)
                # If the rationale is the same, we do not include that threshold.
                # In this way, we will not consider in the average the same omission.
                if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                    id_top = None

                id_tops.append(id_top)      # store indices of most important features for current threshold

                if id_top is None:      # skip threshold if redundant
                    continue

                last_id_top = set(id_top)       # updates last id set
                id_top.sort()

                # Comprehensiveness
                # The only difference between comprehensiveness and sufficiency is the computation of the removal.
                # For the comprehensiveness: we remove the terms in the discrete rationale, with the specified removal type.
                words_removed = [words_transcript[i] for i in id_top]   # words to be removed
                removed_features.append((v, words_removed))
                audio_removed = remove_specified_words(audio, words_removed, removal_type=removal_type)     # modified audio
                audio_removed_np = pydub_to_np(audio_removed)[0]    # audio converted to np for prediction

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label
                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                        :, target[target_class_idx]
                    ][0]
                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)
                used_thresholds.append(v)  # track the threshold that was used

                scores_by_threshold[v].append(removal_importance)

            if removal_importances == []:       # if no removals were performed
                return EvaluationSpeech(self.SHORT_NAME, 0, target)

            # compute AOPC comprehensiveness
            aopc_comprehensiveness = _compute_aopc(removal_importances)     # from ferret documentation: return mean(scores)
            aopc_comprehensiveness_multi_label.append(aopc_comprehensiveness)
            fig = plot_aopc_curve_compr(used_thresholds, removal_importances, title=f"AOPC Comprehensiveness Curve for {labels[target_class_idx]}", file_suffix=labels[target_class_idx])

            removed_features_by_label[target_class_idx] = removed_features  # store removed features for this label

        evaluation_output = EvaluationSpeech(self.SHORT_NAME, aopc_comprehensiveness_multi_label, target)

        labels = self.model_helper.get_text_labels_with_class(target)



        return evaluation_output


class AOPC_Sufficiency_Evaluation_Speech:
    NAME = "aopc_sufficiency"
    SHORT_NAME = "aopc_suff"
    # Lower is better
    BEST_SORTING_ASCENDING = True
    TYPE_METRIC = "faithfulness"

    def __init__(self, model_helper, **kwargs):
        self.model_helper = model_helper

    def compute_evaluation(
        self,
        explanation: ExplanationSpeech,
        target=None,
        words_transcript: List = None,
        phonemization: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        sliding: bool = False,
        **evaluation_args,
    ) -> EvaluationSpeech:
        """Evaluate an explanation on the AOPC Sufficiency metric. Measures how much the prediction changes
        when only the most important features are RETAINED (lower score --> features are sufficient to explain the
        model's decision).
        Extracts the audios and the transcription, computes the model's prediction then retains only the most important
        features (words/phoneme) and removes the rest from the audio and compute the prediction again. Computes the
        difference between and averages it across multiple thresholds to compute the AOPC sufficiency score.

        Args:
            explanation (Explanation): the explanation to evaluate
            target: class labels for which the explanation is evaluated - deprecated
            evaluation_args (dict): arguments for the evaluation

        Returns:
            Evaluation : the AOPC Sufficiency score of the explanation
        """

        # from ferret documentation:
        # only_pos ->  As a default, we consider in the rationale only the terms influencing positively the prediction
        # removal_args -> As a default, we remove from 10% to 100% of the tokens.
        # structure: removal_args = {"remove_tokens": True,"based_on": "perc", "thresholds": np.arange(0.1, 1.1, 0.1)}
        _, only_pos, removal_args, _ = parse_evaluator_args(evaluation_args)

        assert ("perturb_paraling" not in explanation.explainer), f"{explanation.explainer} not supported"

        if "+" in explanation.explainer:
            # The explainer name contain "+" to specify the removal type --> loo_speech+silence
            partial = explanation.explainer.split("+")[1]
            removal_type = partial.split(":")[1]
            # print(removal_type)
        else:
            # Default
            removal_type = "silence"

        if target is not None:
            warnings.warn(
                'The "target" argument is deprecated and will be removed in a future version. The explanation target are used as default.'
            )

        audio_path = explanation.audio_path
        target = explanation.target
        audio = AudioSegment.from_wav(audio_path)  # get the audio from audio_path
        audio_np = pydub_to_np(audio)[0]
        labels = self.model_helper.get_text_labels_with_class(target)

        # Get prediction probability of the input sentence for the target
        ground_truth_probs = self.model_helper.predict([audio_np])
        # print("ground_truth_probs", ground_truth_probs)  # debug print
        # ground_truth_probs (array([[9.25510613e-10, 1.89430551e-07, 1.56134888e-07, 9.99869227e-01,
        # 1.30391098e-04, 8.84956641e-09]]), array([[1.41779232e-12, 8.12982928e-12, 3.92900716e-07, 2.87418607e-05,
        # 9.99970675e-01, 1.92245300e-11, 2.69004175e-07, 5.77682790e-10, 6.08114991e-12, 1.92484900e-11, 2.22094449e-08, 5.43362707e-12,
        # 1.35985334e-10, 6.84768377e-13]]), array([[1.63515005e-03, 5.10979007e-05, 9.97955680e-01, 3.58074176e-04]]))

        # Get the probability of the target classes
        if self.model_helper.n_labels > 1:
            # Multi-label setting - probability of the target classes for each label (list of size = number of labels)
            ground_truth_probs_target = [
                ground_truth_probs[e][:, tidx][0] for e, tidx in enumerate(target)
            ]
        else:
            # Single probability
            ground_truth_probs_target = [ground_truth_probs[0][target[0]]]

        # print(f"ground_truth_probs_target: {ground_truth_probs_target}")  # debug print
        # ground_truth_probs_target: [0.9998692274093628, 0.9999706745147705, 0.9979556798934937]

        # Split the audio into word-level audio segments
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
                sliding=sliding,
            )

        # discrete rationale --> subset of features that are most important based on the explanation score
        get_discrete_rationale_function = (
            _check_and_define_get_id_discrete_rationale_function(
                removal_args["based_on"]  # based on = perc
            )
        )

        # "thresholds": np.arange(0.1, 1.1, 0.1) from the ferret documentation
        thresholds = removal_args["thresholds"]
        score_explanation = explanation.scores

        # In this way, we allow for multi-label explanations
        # We have a list of score_explanations, one for each label
        # Each score explanation has length equal to the number of features: number of words in the case of word-level
        # or phoneme level explanations
        # or 1 in the case of paralinguistic audio level explanation (one paralinguistic feature at the time)
        if score_explanation.ndim == 1:
            score_explanations = [score_explanation]
        else:
            score_explanations = score_explanation
        # print("score_explanations: ", score_explanations)
        # score_explanations:  [[0.34367895 0.22449489 0.3593378  0.6784782  0.8538744  0.17034635
        #   1.6527896  1.4775021  0.2003593  0.54604363 0.36717033 0.56480443
        #   0.5244094  0.18765727 0.30558315 2.2311673 ]
        #  [0.41089773 0.42064685 0.69537    0.899253   1.3516613  0.31251425
        #   2.1858501  2.5494413  0.4076993  1.1006801  0.63760823 1.5279737
        #   1.4699554  0.7930793  0.8575276  2.168511  ]
        #  [0.46315846 0.21265785 0.49409765 0.78953034 0.6146893  0.17657903
        #   1.0855485  2.5159912  0.34199783 0.8116315  0.39372948 0.652283
        #   0.70598906 0.3194802  0.39571747 1.5337318 ]]

        scores_by_threshold = {t: [] for t in thresholds}
        removed_features_by_label = {}
        aopc_sufficiency_multi_label = list()

        # We iterate over the target classes for a multi-label setting
        # In the case of single label, we iterate only once
        for target_class_idx, score_explanation in enumerate(score_explanations):
            removal_importances = list()  # store changes in prediction probability for thresholds
            id_tops = list()  # indices of most important features
            last_id_top = None  # last set of important features
            used_thresholds = list()  # track which thresholds were actually used
            removed_features = []  # track features removed at each threshold

            # Ground truth probabilities of the target label (target_class_idx) and target class (target[target_class_idx]])
            # It is the output probability of the target class itself in the case of single label
            original_prob = ground_truth_probs_target[target_class_idx]  # model prediction for target class

            # We compute the difference in probability for all the thresholds
            for v in thresholds:
                # Get rationale (most important features) from score explanation for the current thresholds
                id_top = get_discrete_rationale_function(score_explanation, v, only_pos)
                # v is converted to an int and words are removed based on its value

                # If the rationale is the same, we do not include that threshold.
                # In this way, we will not consider in the average the same omission.
                if id_top is not None and last_id_top is not None and set(id_top) == last_id_top:
                    id_top = None

                id_tops.append(id_top)  # store indices of most important features for current threshold

                if id_top is None:  # skip threshold if redundant
                    continue

                last_id_top = set(id_top)  # updates last id set
                id_top.sort()

                # Sufficiency
                # The only difference between comprehensiveness and sufficiency is the computation of the removal.
                # For the sufficiency: we keep only the terms in the discrete rationale.
                # Hence, we remove all the other terms.
                words_removed = [
                    words_transcript[i]
                    for i in range(len(words_transcript))
                    if i not in id_top
                ]
                removed_features.append((v, words_removed))
                audio_removed = remove_specified_words(audio, words_removed, removal_type=removal_type)  # modified audio
                audio_removed_np = pydub_to_np(audio_removed)[0]  # audio converted to np for prediction

                # Probability of the modified audio
                audio_modified_probs = self.model_helper.predict([audio_removed_np])

                # Probability of the target class (and label) for the modified audio
                if self.model_helper.n_labels > 1:
                    # In the multi-label setting, we have a list of probabilities for each label
                    # We first take the probability of the corresponding target label target_class_idx
                    # Then we take the probability of the target class for that label target[target_class_idx]
                    modified_prob = audio_modified_probs[target_class_idx][
                                    :, target[target_class_idx]
                                    ][0]
                else:
                    # Single probability
                    # We take the probability of the target class target[target_class_idx]
                    modified_prob = audio_modified_probs[0][target[target_class_idx]]

                # compute probability difference
                removal_importance = original_prob - modified_prob
                removal_importances.append(removal_importance)

                used_thresholds.append(v)  # track the threshold that was used
                scores_by_threshold[v].append(removal_importance)

            if removal_importances == []:  # if no removals were performed
                return EvaluationSpeech(self.SHORT_NAME, 0, target)

            # compute AOPC sufficiency
            aopc_sufficiency = _compute_aopc(removal_importances)
            aopc_sufficiency_multi_label.append(aopc_sufficiency)
            fig = plot_aopc_curve_suff(used_thresholds, removal_importances, title=f"AOPC Suffiency Curve for {labels[target_class_idx]}", file_suffix=labels[target_class_idx])

            removed_features_by_label[target_class_idx] = removed_features  # store removed features for this label

        evaluation_output = EvaluationSpeech(
            self.SHORT_NAME, aopc_sufficiency_multi_label, target
        )

        return evaluation_output
