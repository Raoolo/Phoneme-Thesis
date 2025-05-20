import numpy as np
from typing import Dict, List, Union, Tuple
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import torch
import seaborn as sns
from speechxai.explainers.loo_speech_explainer import LOOSpeechExplainer        # different explainer for speech
from speechxai.explainers.gradient_speech_explainer import GradientSpeechExplainer
from speechxai.explainers.lime_speech_explainer import LIMESpeechExplainer
from speechxai.explainers.paraling_speech_explainer import ParalinguisticSpeechExplainer
from speechxai.explainers.shap_speech_explainer import SHAPSpeechExplainer

## Set seed
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
# If True, We use the add_noise of torch audio
USE_ADD_NOISE_TORCHAUDIO = True
REFERENCE_STR = "-"
SCORES_PALETTE = sns.diverging_palette(240, 10, as_cmap=True)


class Benchmark:
    def __init__(
        self,
        model,
        feature_extractor,
        device: str = "cuda:0",
        language: str = "en",
        explainers=None,
    ):
        # initialize model, feature extractor and device
        self.model = model
        self.feature_extractor = feature_extractor
        self.model.eval()
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        # get the appropriate MODEL HELPER based on the model type
        # the model helper provides a consistent interface for interacting with different models 

        if "superb-ic" in self.model.name_or_path:    # stores the string passed in "from_pretrained(str)""
            # We are using the FSC model - It has three output classes
            from .model_helpers.model_helper_fsc import ModelHelperFSC

            self.model_helper = ModelHelperFSC(
                self.model, self.feature_extractor, self.device, "en"
            )
        elif "ITALIC" in self.model.name_or_path or "italic" in self.model.name_or_path:
            # if using the ITALIC model for italian
            from .model_helpers.model_helper_italic import ModelHelperITALIC
            print("ITALIC")
            self.model_helper = ModelHelperITALIC(
                self.model, self.feature_extractor, self.device, "it"
            )
        elif "slurp" in self.model.name_or_path or "SLURP" in self.model.name_or_path or \
            "massive" in self.model.name_or_path or "MASSIVE" in self.model.name_or_path:
            # if using the ITALIC model for italian
            from .model_helpers.model_helper_italic import ModelHelperITALIC
            print("MASSIVE or SLURP")
            self.model_helper = ModelHelperITALIC(
                self.model, self.feature_extractor, self.device, language,
            )
        else:
            # We are using the ER model - It has one output class
            from .model_helpers.model_helper_er import ModelHelperER

            self.model_helper = ModelHelperER(
                self.model, self.feature_extractor, self.device, language
            )

        # all into the explainers folder
        if explainers is None:
            # Use the default explainers (dictionary)
            self.explainers = {
                "LOO": LOOSpeechExplainer(self.model_helper),
                "Gradient": GradientSpeechExplainer(
                    self.model_helper, multiply_by_inputs=False
                ),
                "GradientXInput": GradientSpeechExplainer(
                    self.model_helper, multiply_by_inputs=True
                ),
                "LIME": LIMESpeechExplainer(self.model_helper),
                "perturb_paraling": ParalinguisticSpeechExplainer(self.model_helper),
                "SHAP" : SHAPSpeechExplainer(self.model_helper),
            }

    def set_explainers(self, explainers):
        # allows for custom explainers
        self.explainers = explainers

    def predict(
        self,
        audios: List[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # TMP (temporary?)
        # Just a wrapper around ModelHelperFSC.predict/ModelHelperFSC.predict_single We use the second to overcome the padding issue
        return self.model_helper.predict(audios)
      

    def explain(
        self,
        audio_path: str,
        target_class: str = None,
        methodology: str = "LOO",
        phonemization: bool = False,
        word_paraling: bool = False,  # Used to activate word-level for paralinguistic features
        words_transcript: List = None,
        perturbation_types: List[str] = [
            "pitch shifting",
            "pitch shifting down",
            "pitch shifting up",
            "time stretching",
            "time stretching down",
            "time stretching up",
            "reverberation",
            "noise",
            "degradation",
            "white noise",
            "pink noise",
            "claps",
        ],
        removal_type: str = "silence",  # Used only for LOO and LIME
        aggregation: str = "mean",  # Used only for Gradient and GradientXInput
        window_size: int = None,   # Used for aggregating phonemes
        respect_word_boundaries: bool = True,
        num_samples: int = 1000,  # Used only for LIME
        single_perturbation_value: float = None,
        verbose: bool = False,
        verbose_target: int = 0,
        display_audio: bool = False,
        complete_perturbation: bool = False,
        perturbations_list: list = None,     # for paraling
        visualization: bool = False,
        sliding: bool = False,
    ):
        """
        Explain the prediction of the model.
        Returns the importance of each segment in the audio.
        """
        explainer_args = {}
        perturbed_audios = None
        ## Get the importance of each class (action, object, location) according to the perturb_paraling type
        if methodology == "perturb_paraling":
            explainer = self.explainers["perturb_paraling"]
            allowed_values = {
                "pitch shifting", "pitch shifting down", "pitch shifting up", "reverberation",
                "time stretching", "time stretching down", "time stretching up", "stress",
                "degradation", "noise", "white noise", "pink noise", "claps", "intensity",}

            assert set(perturbation_types).issubset(allowed_values), \
                f"Invalid values found: {set(perturbation_types) - allowed_values}"

            if phonemization and word_paraling:
                raise ValueError(
                    "Cannot use both phoneme level and word level at the same time."
                )

            if phonemization or word_paraling:
                if len(perturbation_types) > 1:
                    print(f"Using only the first type of perturbation: {perturbation_types[0]}")
                explanations, perturbed_audios = explainer.compute_explanation(
                    audio_path=audio_path,
                    target_class=target_class,
                    perturbation_type=perturbation_types[0],
                    verbose=verbose,
                    verbose_target=verbose_target,
                    single_perturbation_value=single_perturbation_value,
                    phonemization=phonemization,
                    word_level=word_paraling,
                    complete_perturbation=complete_perturbation,
                    window_size=window_size,
                    respect_word_boundaries=respect_word_boundaries,
                    perturbations_list=perturbations_list,
                    sliding=sliding,
                )
                return explanations, perturbed_audios
            else:
                explanations = []
                for perturbation_type in perturbation_types:
                    # call the explainer and compute the explanation for each perturbation of the paraling
                    explanation, perturbed_audios = explainer.compute_explanation(
                        audio_path=audio_path,
                        target_class=target_class,
                        perturbation_type=perturbation_type,
                        verbose=verbose,
                        verbose_target=verbose_target,
                        perturbations_list=perturbations_list,
                    )
                    explanations.append(explanation)
                # table = self.create_table(importances)
                return explanations, perturbed_audios
        else:
            if methodology not in self.explainers:
                raise ValueError(
                    f'Explainer {methodology} not supported. Choose between "LOO", "Gradient", "GradientXInput", "LIME", "SHAP, "perturb_paraling"'
                )
            if "LOO" in methodology:
                explainer_args["removal_type"] = removal_type
                explainer_args["single_perturbation_value"] = single_perturbation_value
                explainer_args["complete_perturbation"] = complete_perturbation
                explainer_args["perturbation_list"] = perturbations_list
                explainer_args["verbose"] = verbose
            elif "LIME" in methodology or "SHAP" in methodology:
                explainer_args["removal_type"] = removal_type
                explainer_args["num_samples"] = num_samples
            else:
                explainer_args["aggregation"] = aggregation

            explainer = self.explainers[methodology]
            explanation = explainer.compute_explanation(
                audio_path=audio_path,
                target_class=target_class,
                words_transcript=words_transcript,
                phonemization = phonemization,
                window_size = window_size,
                respect_word_boundaries = respect_word_boundaries,
                visualization = visualization,
                sliding = sliding,
                **explainer_args,
            )

            return explanation

        ## Debug print
        # print(f"Generated explanations: {explanations}")
        ## Return table of explanations


    def global_explanations(
        self,
        audio_paths: List[str],
        target_class: str = None,
        methodology: str = "LOO",
        aggregate: bool = False,
        **explainer_kwargs,
    ) -> Union[Dict[str, float], Dict[str, Dict[str, float]]]:
        """
        Compute global explanations by averaging instance-level importance scores
        over a list of audio files.
        """

        # dictionary to keep aggregated scores, for multi label I use a double nested dictionary
        global_scores = {}
        explanations = []

        for audio_path in audio_paths:
            # get the explanation for each audio
            explanation = self.explain(
                audio_path, target_class=target_class, methodology=methodology, **explainer_kwargs
            )

            if not explanation:
                print(f"No valid transcription found for {audio_path}. Skipping audio.")
                continue
            explanations.append(explanation)

            if aggregate:
                if self.model_helper.n_labels > 1:
                    # multi label
                    label_names = self.model_helper.get_text_labels_with_class(explanation.target)
                    # iterate over each label
                    for j, label in enumerate(label_names):
                        if label not in global_scores:      # create key if not present already
                            global_scores[label] = {}
                        for i, phoneme in enumerate(explanation.features):  # for each phoneme
                            score = explanation.scores[j, i]        # take the score associated
                            if phoneme not in global_scores[label]:     # create phoneme key if not present
                                global_scores[label][phoneme] = []
                            global_scores[label][phoneme].append(score)     # append the score
                else:
                    # single label
                    for i, phoneme in enumerate(explanation.features):
                        score = explanation.scores[i]
                        if phoneme not in global_scores:
                            global_scores[phoneme] = []
                        global_scores[phoneme].append(score)

            #print(f"Global explanations: {global_scores}")      # debug print
            # print(f"Global explanations: {len(global_scores)}")

            # compute average score for each phoneme
            if self.model_helper.n_labels > 1:
                avg_scores = {}
                for label, phoneme_scores in global_scores.items():
                    avg_scores[label] = {phoneme: np.mean(scores) for phoneme, scores in phoneme_scores.items()}
            else:
                avg_scores = {phoneme: np.mean(scores) for phoneme, scores in global_scores.items()}

            # print(f"Global explanations: {len(avg_scores)}")        # debug print



        return avg_scores, explanations

    def create_table(
        self,
        explanations,
        axis=1,  # append the scores to the columns
        phonemization: bool = False,
    ) -> pd.DataFrame:
        """
        Create a dataframe from a list of explanations
        Args:
            explanations: list of explanations or single explanation
            axis: 0 for appending the scores by rows, 1 for appending by columns
        Creates a table with the words or paralinguistic feature(s),
        and the difference p(y|x\F) - p(y|x) for each class.
        """

        if type(explanations) == list:
            if axis == 1:
                # Append the scores as columns, the code handles a list of explanations
                # where each explanation corresponds to a single feature for the same target class

                # We have a list of explanations for the same target class, each for a single feature
                assert [    # all explanations are for the same target class
                    False
                    for i in range(0, len(explanations) - 1)
                    if explanations[i].target != explanations[i + 1].target
                ] == [], "The explanations must have the same target class"

                # for explanation in explanations:
                    # print(f"Explanation.scores dimensions: {explanation.scores.shape}")
                    # print(f"Explanation.scores content: {explanation.scores}")

                importance_df = pd.DataFrame([explanation.scores for explanation in explanations]).T

                # use features from first explanations as column
                importance_df.columns = [explanation.features[0] for explanation in explanations]

                # We take the target of the first explanation. We know that all the explanations have the same target
                target_class_names = self.model_helper.get_text_labels_with_class(
                    explanations[0].target
                )   # takes the text labels associated with the target class using the helper model

                # and then sets these labels as the index of the dataframe
                importance_df.index = (
                    list(target_class_names)
                    if self.model_helper.n_labels > 1  # multilabel
                    else [target_class_names]  # single label
                ) #Check if all explanations are for the same target class

        else:   # single explanation object
            explanation = explanations
            # print(f"Explanation.score: {explanation.scores}")
            importance_df = pd.DataFrame(explanation.scores)
            importance_df.columns = explanation.features
            target_class_names = self.model_helper.get_text_labels_with_class(
                explanation.target
            )
            importance_df.index = (
                list(target_class_names)
                if self.model_helper.n_labels > 1  # multilabel
                else [target_class_names]  # single label
            )
        return importance_df

    def show_table(self, explanations, apply_style: bool = True, decimals=4):
        # Rename duplicate columns (tokens) by adding a suffix
        table = self.create_table(explanations)

        # if sum(table.columns.duplicated().astype(int)) > 0:
        #     table.columns = pd.io.parsers.base_parser.ParserBase(
        #         {"names": table.columns, "usecols": None}
        #     )._maybe_dedup_names(table.columns)

        # changed the method to avoid using a parser and a private method which was removed from the library
        if table.columns.duplicated().any():
            # if column already exists append an affix at the end of the name
            columns = table.columns
            seen = {}
            new_columns = []
            for col in columns:
                if col in seen:
                    seen[col] += 1
                    new_columns.append(f"{col}_{seen[col]}")
                else:
                    seen[col] = 0
                    new_columns.append(col)
            table.columns = new_columns

        # new method to actually use the apply_stile parameter
        if apply_style:
            return (
                table.apply(pd.to_numeric)
                .style.background_gradient(axis=1, cmap=SCORES_PALETTE, vmin=-1, vmax=1)
                .format(precision=decimals)
            )
        else:
            return table.apply(pd.to_numeric).style.format(precision=decimals)



    def explain_variations(self, audio_path, perturbation_types, target_class=None):
        perturbation_df_by_type = self.explainers["perturb_paraling"].explain_variations(audio_path, perturbation_types, target_class)
        return perturbation_df_by_type

    def plot_variations(self, perturbation_df_by_type, show_diff=False, figsize=(5, 5)):
        """
        perturbation_df_by_type: dictionary of dataframe
        show_diff: if True, show the difference with the baseline
        """

        plt.rcParams.update(
            {
                "text.usetex": False,
            }
        )

        fig, axs = plt.subplots(len(perturbation_df_by_type), 1, figsize=figsize)

        for e, (perturbation_type, perturbation_df) in enumerate(
            perturbation_df_by_type.items()
        ):
            ax = axs[e]

            prob_variations_np = perturbation_df.values

            if show_diff:
                reference_value = REFERENCE_STR

                prob_variations_np = (
                    perturbation_df[reference_value].values.reshape(-1, 1)
                    - prob_variations_np
                )

            target_classes_show = list(perturbation_df.index)

            x_labels = list(perturbation_df.columns)
            label_size = 11
            if show_diff:
                norm = mcolors.TwoSlopeNorm(vmin=-1, vmax=1, vcenter=0)
                cmap = plt.cm.PiYG.reversed()
            else:
                norm = mcolors.TwoSlopeNorm(vmin=0, vmax=1, vcenter=0.5)
                cmap = plt.cm.Purples
            im = ax.imshow(prob_variations_np, cmap=cmap, norm=norm, aspect="auto")
            ax.set_yticks(
                np.arange(len(target_classes_show)),
                labels=target_classes_show,
                fontsize=label_size,
            )

            if len(x_labels) > 10:
                if (
                    perturbation_type == "time stretching"
                    or perturbation_type == "noise"
                ):
                    x_labels = [
                        x_label if ((x_label == REFERENCE_STR) or (e % 3 == 0)) else ""
                        for e, x_label in enumerate(x_labels)
                    ]
                else:
                    x_labels = [
                        x_label if ((x_label == REFERENCE_STR) or (e % 4 == 0)) else ""
                        for e, x_label in enumerate(x_labels)
                    ]

            x_labels = [
                r"x" if (x_label == REFERENCE_STR) else x_label for x_label in x_labels
            ]

            if perturbation_type == "time stretching":
                x_labels[0] = str(x_labels[0]) + "\nfaster"
                x_labels[-1] = str(x_labels[-1]) + "\nslower"

                ax.set_xlabel("stretching factor", fontsize=label_size, labelpad=-2)
            elif perturbation_type == "pitch shifting":
                x_labels[0] = str(x_labels[0]) + "\nlower"
                x_labels[-1] = str(x_labels[-1]) + "\nhigher"
                ax.set_xlabel("semitones", fontsize=label_size, labelpad=-2)
            elif perturbation_type == "noise":
                x_labels[-1] = str(x_labels[-1]) + "\nnoiser"
                ax.set_xlabel(
                    "signal-to-noise ratio (dB)", fontsize=label_size, labelpad=-2
                )
            ax.set_xticks(
                np.arange(len(x_labels)), labels=x_labels, fontsize=label_size
            )

            ax.set_title(perturbation_type, fontsize=label_size)

            for lab in ax.get_xticklabels():
                if lab.get_text() == "x":
                    lab.set_fontweight("bold")
        plt.tight_layout()
        plt.subplots_adjust(hspace=2.3)
        cbar = fig.colorbar(
            im,
            ax=axs.ravel().tolist(),
            location="right",
            anchor=(0.3, 0.3),  # 0.3
            shrink=0.7,
        )
        cbar.ax.tick_params(labelsize=label_size)

        plt.show()
        return fig
