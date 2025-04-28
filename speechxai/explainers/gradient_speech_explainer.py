from speechxai.explainers.explanation_speech import ExplanationSpeech
from speechxai.explainers.utils_removal import transcribe_audio
from speechxai.utils import pydub_to_np
from typing import List
from pydub import AudioSegment
from captum.attr import Saliency, InputXGradient
import numpy as np
import torch


class GradientSpeechExplainer:
    NAME = "Gradient"

    def __init__(self, model_helper, multiply_by_inputs: bool = False):
        self.model_helper = model_helper
        self.multiply_by_inputs = multiply_by_inputs     # if false -> saliency, if true -> gradient

        if self.multiply_by_inputs:
            self.NAME += " (x Input)"

    def _get_gradient_importance_frame_level(
        self, audio, target_class, target_label=None
    ):
        """
        Compute the gradient importance for each frame of the audio w.r.t. the target class.
        Args:
            audio: audio - np.array
            target_class: target class - int
            target_label: target label - int - Used only in a multilabel scenario as for FSC
        """
        torch.set_grad_enabled(True)  # enables gradient computation

        # Function to compute the logits
        if self.model_helper.n_labels > 1:
            # We get the logits for the given label
            func = self.model_helper.get_logits_function_from_input_embeds_by_label(target_label)
        else:
            func = self.model_helper.get_logits_from_input_embeds

        # if saliency: It returns the gradients with respect to inputs, returning the absolute value of the gradients.
        # These gradients indicate how sensitive the output si to changes in the input
        # if inputxGradient: It multiplies input with the gradient with respect to input, emphasizes features
        # that are both important (high gradient) and have large magnitude (high input value)
        dl = InputXGradient(func) if self.multiply_by_inputs else Saliency(func)

        # extract features
        inputs = self.model_helper.feature_extractor(
            [audio_i.squeeze() for audio_i in [audio]],
            sampling_rate=self.model_helper.feature_extractor.sampling_rate,
            padding=True,
            return_tensors="pt",
        )
        # print(f"inputs value: {inputs}")
        print(inputs.input_values)
        print(inputs.input_values.min(), inputs.input_values.max())

        # computes the attributions for the input features
        input_len = inputs["attention_mask"].sum().item()
        attr = dl.attribute(inputs.input_values, target=target_class)
        # print(f"attr value (gradients or gradientsxinput): {attr}")
        attr = attr[0, :input_len].detach().cpu()

        # pool over hidden size
        attr = attr.numpy()
        return attr



    def compute_explanation(
        self,
        audio_path: str,
        target_class=None,
        words_transcript: List = None,
        no_before_span: bool = True,
        aggregation: str = "mean",
        phonemization: bool = False,
        window_size: int = None,
        respect_word_boundaries: bool = True,
        visualization: bool = False,
        sliding: bool = False,
    ) -> ExplanationSpeech:
        """
        Compute the word-level explanation for the given audio.
        Args:
        audio_path: path to the audio file
        target_class: target class - int - If None, use the predicted class
        no_before_span: if True, it also considers the span before the word. This is because we observe gradient give importance also for the frame just before the word
        aggregation: aggregation method for the frames of the word. Can be "mean" or "max"
        """

        if aggregation not in ["mean", "max", "sum"]:
            raise ValueError(
                "Aggregation method not supported, choose between 'mean', 'max' and 'sum'"
            )

        # Load audio and convert to np.array
        audio = pydub_to_np(AudioSegment.from_wav(audio_path))[0]

        # Predict logits/probabilities
        logits_original = self.model_helper.predict([audio])

        # Check if single label or multilabel scenario as for FSC then determine tha classes
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

        if words_transcript is None:
            # Transcribe audio at word/phoneme level
            _, words_transcript = transcribe_audio(
                audio_path=audio_path, language=self.model_helper.language,
                phonemization=phonemization, window_size=window_size, respect_word_boundaries=respect_word_boundaries,
                sliding=sliding)

        # Compute gradient importance for each frame for each label
        scores = []
        for target_label, target_class in enumerate(targets):
            # Get gradient importance for each frame
            attr = self._get_gradient_importance_frame_level(audio, target_class, target_label)

            old_start = 0
            old_start_ms = 0
            features = []
            importances = []
            a, b = 5, 5  # 50, 20

            # map the frame-level gradients to ach word/phoneme
            for word in words_transcript:
                if no_before_span:
                    # We directly consider the transcribed word/phoneme
                    start_ms = (word["start"] * 1000 - a) / 1000
                    end_ms = (word["end"] * 1000 + b) / 1000
                else:
                    # We also include the frames before the word/phoneme
                    start_ms = old_start_ms
                    end_ms = (word["end"] * 1000) / 1000

                start = int(start_ms * self.model_helper.feature_extractor.sampling_rate)
                end = int(end_ms * self.model_helper.feature_extractor.sampling_rate)

                # print(f"attributes: {attr[start:end]}")
                # Slice of the importance for the given word/phoneme
                word_importance = attr[start:end]

                # Consider also the spans between words, used for prosody
                # #span_before = attr[old_start:start]

                if aggregation == "max":    # highlight the most important features
                    word_importance = np.max(word_importance)
                elif aggregation == "sum":  # to get cumulative importance
                    word_importance = np.sum(word_importance)
                else:       # balance importance through feature
                    word_importance = np.mean(word_importance)

                old_start = end
                old_start_ms = end_ms
                importances.append(word_importance)
                if phonemization:
                    features.append(word["char"])
                else:
                    features.append(word["word"])

            # Consider also the spans between words
            # importances.append(np.mean(span_before))
            # features.append('-')

            # Consider also the spans between words
            # Final span
            # final_span = attr[old_start:len(audio_np)]
            # features.append('-')

            # if aggregation == "max":
            #    importances.append(np.max(final_span))
            # else:
            #    importances.append(np.mean(final_span))
            scores.append(np.array(importances))

        if n_labels > 1:
            # Multilabel scenario as for FSC
            scores = np.array(scores)
        else:
            scores = np.array([importances])



        if phonemization:
            features = [word["char"] for word in words_transcript]
        else:
            features = [word["word"] for word in words_transcript]


        explanation = ExplanationSpeech(
            features=features,
            scores=scores,
            explainer=f"{self.NAME}+aggregation:{aggregation}+window:{window_size}",
            target=targets if n_labels > 1 else targets,
            audio_path=audio_path,
        )

        return explanation
