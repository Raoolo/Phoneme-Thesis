import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd
from typing import List, Dict
import seaborn as sns
from pydub import AudioSegment
from IPython.display import display
import os
from speechxai.utils import pydub_to_np, print_log

plot_dir = "./plots/"
os.makedirs(plot_dir, exist_ok=True)

def visualize_whisperx_results(segments, phonemization: bool = False, og_text: str = None):

    fig, ax = plt.subplots(figsize=(12, 6))

    segment_color = 'lightblue'
    word_color = 'lightgreen'
    char_color = 'lightcoral'
    silence_color = 'lightgray'

    if phonemization:
        y_levels = {'segment': 2, 'word': 1, 'char': 0}
        y_text = {'segment': 2.5, 'word': 1.5, 'char': 0.5}
        max_y = 3
    else:
        y_levels = {'segment': 1, 'word': 0}
        y_text = {'segment': 1.5, 'word': 0.5}
        max_y = 2

    all_times = []
    for segment in segments:
        all_times.extend([segment['start'], segment['end']])
    start_time = min(all_times)
    end_time = max(all_times)

    ax.add_patch(patches.Rectangle(
        (start_time - 0.1, 0),
        end_time - start_time + 0.2,
        max_y,
        facecolor=silence_color,
        edgecolor='none'
    ))

    for segment in segments:
        segment_start = segment['start']
        segment_end = segment['end']
        ax.add_patch(patches.Rectangle(
            (segment_start, y_levels['segment']),
            segment_end - segment_start,
            1,
            edgecolor='blue',
            facecolor=segment_color,
            label='Segment' if segment == segments[0] else ""
        ))
        ax.text(
            segment_start + (segment_end - segment_start) / 2,
            y_text['segment'],
            segment['text'],
            ha='center',
            va='center'
        )

        for word in segment['words']:
            if 'start' in word and 'end' in word:
                ax.add_patch(patches.Rectangle(
                    (word['start'], y_levels['word']),
                    word['end'] - word['start'],
                    1,
                    edgecolor='green',
                    facecolor=word_color,
                    label='Word' if word == segment['words'][0] and segment == segments[0] else ""
                ))
                ax.text(
                    word['start'] + (word['end'] - word['start']) / 2,
                    y_text['word'],
                    word['word'],
                    ha='center',
                    va='center'
                )

        if phonemization:
            for char in segment['chars']:
                if 'start' in char and 'end' in char:
                    ax.add_patch(patches.Rectangle(
                        (char['start'], y_levels['char']),
                        char['end'] - char['start'],
                        1,
                        edgecolor='red',
                        facecolor=char_color,
                        label='Phoneme' if char == segment['chars'][0] and segment == segments[0] else ""
                    ))
                    ax.text(
                        char['start'] + (char['end'] - char['start']) / 2,
                        y_text['char'],
                        char['char'],
                        ha='center',
                        va='center'
                    )

    ax.set_xlabel('Time (seconds)')
    if phonemization:
        ax.set_yticks([0.5, 1.5, 2.5])
        ax.set_yticklabels(['Phonemes', 'Words', 'Segment'])
        ax.set_ylim(0, 3)
    else:
        ax.set_yticks([0.5, 1.5])
        ax.set_yticklabels(['Words', 'Segment'])
        ax.set_ylim(0, 2)

    ax.set_xlim(start_time - 0.1, end_time + 0.1)
    ax.legend(loc='upper right')

    title = 'Transcription Visualization'
    if og_text:
        title += f'\nOriginal Text: "{og_text}"'

    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "alignment_results"))
    plt.show()


def plot_loo_importance(audio: np.ndarray, sampling_rate: int, segments: List[Dict],
                        scores: np.ndarray, phonemization: bool = False, targets: List = None) -> plt.Figure:

    # multi-label case
    if scores.ndim > 1:
        n_labels = scores.shape[0]
        fig = plt.figure(figsize=(15, 4 * n_labels))
        gs = fig.add_gridspec(n_labels, 1, height_ratios=[1] * n_labels, hspace=0.4)
        axes = [fig.add_subplot(gs[i]) for i in range(n_labels)]
        if n_labels == 1:
            axes = [axes]
    else:
        fig, axes = plt.subplots(1, 1, figsize=(15, 5))
        axes = [axes]
        scores = scores.reshape(1, -1)

    time = np.arange(len(audio)) / sampling_rate
    max_amplitude = np.max(np.abs(audio))

    for label_idx, ax in enumerate(axes):
        ax.plot(time, audio, color='gray', alpha=0.5, label='Waveform')

        label_scores = scores[label_idx]
        cmap = plt.cm.RdYlBu_r
        norm = plt.Normalize(vmin=np.min(label_scores), vmax=np.max(label_scores))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

        for idx, segment in enumerate(segments):
            start_time = segment['start']
            end_time = segment['end']
            score = label_scores[idx]
            ax.axvspan(start_time, end_time,
                       alpha=0.3,
                       color=cmap(norm(score)))
            text = segment['char'] if phonemization else segment['word']
            ax.text((start_time + end_time) / 2, max_amplitude * 1.4,
                    text, horizontalalignment='center')

        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Importance Score', rotation=270, labelpad=15)
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{targets[label_idx]}', pad=30)  # pad for space for phonemes

        ax.set_ylim(bottom=-max_amplitude * 1.1, top=max_amplitude * 1.3)

    axes[-1].set_xlabel('Time (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "loo_results"))
    return fig


def plot_lime_heatmap(explanation, targets_labels):

    n_labels = explanation.scores.shape[0]
    features = explanation.features
    scores = explanation.scores

    fig, axes = plt.subplots(1 ,1, figsize=(12, 8), gridspec_kw={'height_ratios': [1]})

    ax_heat = axes if isinstance(axes, plt.Axes) else axes[0]

    sns.heatmap(scores, ax=ax_heat, cmap='RdYlGn', center=0,
                xticklabels=features, yticklabels=[f'{targets_labels[i]}' for i in range(n_labels)],
                annot=True, fmt='.2f', cbar_kws={'label': 'LIME Score'})

    ax_heat.set_xticklabels(features, rotation=45, ha='right')
    ax_heat.set_title('LIME Feature Importance Heatmap')

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "lime_results"))

    return fig


def plot_aopc_curve_compr(thresholds, removal_importances, title="AOPC Curve", file_suffix=""):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, removal_importances, marker='o', linestyle='-', color='b')
    plt.xlabel('Thresholds (Percentage of Features Removed)')
    plt.ylabel('Change in Prediction Probability')
    plt.title(title)
    plt.grid(True)

    plt.savefig(os.path.join(plot_dir, f"aopc_results_{file_suffix}"))

    plt.show()

def plot_aopc_curve_suff(thresholds, removal_importances, title="AOPC Curve", file_suffix=""):
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, removal_importances, marker='o', linestyle='-', color='b')
    plt.xlabel('Thresholds (Percentage of Features Retained)')
    plt.ylabel('Change in Prediction Probability')
    plt.title(title)
    plt.grid(True)

    plt.savefig(os.path.join(plot_dir, f"aopc_results_{file_suffix}"))

    plt.show()

def plot_phoneme_importance_heatmap(features, scores, perturbation_values, target):

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        scores,
        xticklabels=perturbation_values,
        yticklabels=features,
        cmap="viridis",
    )
    plt.title(f"Phoneme Importance Across Perturbation Values ({target})")
    plt.xlabel("Perturbation Value")
    plt.ylabel("Phoneme")

    plt.savefig(os.path.join(plot_dir, "heatmap_results"))

    plt.show()

def plot_perturbation_trends(features, scores, perturbation_values, target):

    plt.figure(figsize=(10, 6))
    for i, feature in enumerate(features):
        plt.plot(perturbation_values, scores[i], label=feature)
    plt.title(f"Model Confidence vs. Perturbation Strength ({target})")
    plt.xlabel("Perturbation Value")
    plt.ylabel("Prediction Difference")
    plt.legend()

    plt.savefig(os.path.join(plot_dir, "perturbation_results"))

    plt.show()

def plot_phoneme_importance_bar(features, scores, labels):

    n_labels = scores.shape[0]
    fig, axes = plt.subplots(n_labels, 1, figsize=(10, 6 * n_labels))

    if n_labels == 1:
        axes = [axes]  # axes is a list even for a single label

    for i, ax in enumerate(axes):
        ax.bar(features, scores[i])
        ax.set_title(f"Phoneme Importance Scores ({labels[i]})")
        ax.set_xlabel("Phoneme")
        ax.set_ylabel("Importance Score")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    plt.savefig(os.path.join(plot_dir, "bar_results"))
    plt.show()

def tmp_log_show_info(
    perturbation_type: str,
    perturbation_value: float,
    perturbated_audio: np.ndarray,
    verbose_target: int,
    model_helper: None,
):
    from IPython.display import Audio

    # Display the perturbed audio and show its info for a single class
    # For multi label scenario, we show it for a single class: verbose_target
    # Note that in a single label scenario, verbose_target is ignored (always 0)

    print_log(perturbation_type, perturbation_value)
    # Prediction probability
    predictions = model_helper.predict([perturbated_audio])

    # Predicted label (idx)
    predicted_labels = [v.argmax() for v in predictions]

    # Predicted label (text)
    preds = model_helper.get_text_labels(predicted_labels)

    if model_helper.n_labels > 1:
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


