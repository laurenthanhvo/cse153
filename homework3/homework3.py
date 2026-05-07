# Homework 3: Symbolic Music Generation Using Markov Chains

import random
from glob import glob
from collections import defaultdict
from functools import lru_cache

import numpy as np
from numpy.random import choice

from symusic import Score
from miditok import REMI, TokenizerConfig
from midiutil import MIDIFile

random.seed(42)
midi_files = glob('PDMX_subset/*.mid')
config = TokenizerConfig(num_velocities=1, use_chords=False, use_programs=False)
tokenizer = REMI(config)

duration2length = {
    '0.2.8': 2,   # sixteenth note, 0.25 beat in 4/4 time signature
    '0.4.8': 4,   # eighth note, 0.5 beat in 4/4 time signature
    '1.0.8': 8,   # quarter note, 1 beat in 4/4 time signature
    '2.0.8': 16,  # half note, 2 beats in 4/4 time signature
    '4.0.4': 32,  # whole note, 4 beats in 4/4 time signature
}

EPS = 1e-12

@lru_cache(maxsize=None)
def _tokens_for_file(midi_file):
    midi = Score(midi_file)
    return tuple(tokenizer(midi)[0].tokens)


def _normalize_counts(counts):
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: value / total for key, value in counts.items()}


def _sample_from_distribution(states, probabilities):
    if not states:
        return None
    return random.choices(list(states), weights=list(probabilities), k=1)[0]


def _lookup_probability(transitions, probabilities, key, value, default=EPS):
    if key not in transitions:
        return default
    try:
        index = transitions[key].index(value)
    except ValueError:
        return default
    return probabilities[key][index]


def _beat_length_unigram_probability(midi_files):
    counts = defaultdict(int)
    for midi_file in midi_files:
        for _, beat_length in beat_extraction(midi_file):
            counts[beat_length] += 1
    return _normalize_counts(counts)


# Q1a
def note_extraction(midi_file):
    notes = []
    for token in _tokens_for_file(midi_file):
        if token.startswith('Pitch_'):
            notes.append(int(token.split('_')[1]))
    return notes


# Q1b
def note_frequency(midi_files):
    note_counts = defaultdict(int)
    for midi_file in midi_files:
        for note in note_extraction(midi_file):
            note_counts[note] += 1
    return dict(note_counts)


# Q2
def note_unigram_probability(midi_files):
    note_counts = note_frequency(midi_files)
    unigramProbabilities = {}

    total_notes = sum(note_counts.values())
    if total_notes == 0:
        return unigramProbabilities

    for note, count in note_counts.items():
        unigramProbabilities[note] = count / total_notes

    return unigramProbabilities


# Q3a
def note_bigram_probability(midi_files):
    bigramTransitions = defaultdict(list)
    bigramTransitionProbabilities = defaultdict(list)

    bigram_counts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for previous_note, next_note in zip(notes[:-1], notes[1:]):
            bigram_counts[previous_note][next_note] += 1

    for previous_note, next_counts in bigram_counts.items():
        total = sum(next_counts.values())
        for next_note, count in next_counts.items():
            bigramTransitions[previous_note].append(next_note)
            bigramTransitionProbabilities[previous_note].append(count / total)

    return bigramTransitions, bigramTransitionProbabilities


# Q3b
def sample_next_note(note):
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    if note in bigramTransitions and len(bigramTransitions[note]) > 0:
        return _sample_from_distribution(bigramTransitions[note], bigramTransitionProbabilities[note])

    unigramProbabilities = note_unigram_probability(midi_files)
    return _sample_from_distribution(unigramProbabilities.keys(), unigramProbabilities.values())


# Q4
def note_bigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)

    notes = note_extraction(midi_file)
    if len(notes) == 0:
        return float('inf')

    log_probability_sum = 0.0
    for i, note in enumerate(notes):
        if i == 0:
            probability = unigramProbabilities.get(note, EPS)
        else:
            probability = _lookup_probability(
                bigramTransitions,
                bigramTransitionProbabilities,
                notes[i - 1],
                note,
            )
        log_probability_sum += np.log(max(probability, EPS))

    return np.exp(-log_probability_sum / len(notes))


# Q5a
def note_trigram_probability(midi_files):
    trigramTransitions = defaultdict(list)
    trigramTransitionProbabilities = defaultdict(list)

    trigram_counts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        notes = note_extraction(midi_file)
        for i in range(2, len(notes)):
            key = (notes[i - 2], notes[i - 1])
            trigram_counts[key][notes[i]] += 1

    for key, next_counts in trigram_counts.items():
        total = sum(next_counts.values())
        for next_note, count in next_counts.items():
            trigramTransitions[key].append(next_note)
            trigramTransitionProbabilities[key].append(count / total)

    return trigramTransitions, trigramTransitionProbabilities


# Q5b
def note_trigram_perplexity(midi_file):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)

    notes = note_extraction(midi_file)
    if len(notes) == 0:
        return float('inf')

    log_probability_sum = 0.0
    for i, note in enumerate(notes):
        if i == 0:
            probability = unigramProbabilities.get(note, EPS)
        elif i == 1:
            probability = _lookup_probability(
                bigramTransitions,
                bigramTransitionProbabilities,
                notes[i - 1],
                note,
            )
        else:
            probability = _lookup_probability(
                trigramTransitions,
                trigramTransitionProbabilities,
                (notes[i - 2], notes[i - 1]),
                note,
            )
        log_probability_sum += np.log(max(probability, EPS))

    return np.exp(-log_probability_sum / len(notes))


# Q6
def beat_extraction(midi_file):
    beats = []
    tokens = _tokens_for_file(midi_file)

    current_position = None
    for token in tokens:
        if token.startswith('Position_'):
            current_position = int(token.split('_')[1])
        elif token.startswith('Duration_') and current_position is not None:
            duration_value = token.split('_', 1)[1]
            if duration_value in duration2length:
                beats.append((current_position, duration2length[duration_value]))
            current_position = None

    return beats


# Q7
def beat_bigram_probability(midi_files):
    bigramBeatTransitions = defaultdict(list)
    bigramBeatTransitionProbabilities = defaultdict(list)

    bigram_counts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        beat_lengths = [beat_length for _, beat_length in beat_extraction(midi_file)]
        for previous_beat_length, beat_length in zip(beat_lengths[:-1], beat_lengths[1:]):
            bigram_counts[previous_beat_length][beat_length] += 1

    for previous_beat_length, next_counts in bigram_counts.items():
        total = sum(next_counts.values())
        for beat_length, count in next_counts.items():
            bigramBeatTransitions[previous_beat_length].append(beat_length)
            bigramBeatTransitionProbabilities[previous_beat_length].append(count / total)

    return bigramBeatTransitions, bigramBeatTransitionProbabilities


# Q8a
def beat_pos_bigram_probability(midi_files):
    bigramBeatPosTransitions = defaultdict(list)
    bigramBeatPosTransitionProbabilities = defaultdict(list)

    position_counts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        for beat_position, beat_length in beat_extraction(midi_file):
            position_counts[beat_position][beat_length] += 1

    for beat_position, length_counts in position_counts.items():
        total = sum(length_counts.values())
        for beat_length, count in length_counts.items():
            bigramBeatPosTransitions[beat_position].append(beat_length)
            bigramBeatPosTransitionProbabilities[beat_position].append(count / total)

    return bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities


# Q8b
def beat_bigram_perplexity(midi_file):
    bigramBeatTransitions, bigramBeatTransitionProbabilities = beat_bigram_probability(midi_files)
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    beatUnigramProbabilities = _beat_length_unigram_probability(midi_files)

    beats = beat_extraction(midi_file)
    if len(beats) == 0:
        return float('inf'), float('inf')

    log_probability_sum_Q7 = 0.0
    log_probability_sum_Q8 = 0.0

    for i, (beat_position, beat_length) in enumerate(beats):
        if i == 0:
            probability_Q7 = beatUnigramProbabilities.get(beat_length, EPS)
        else:
            previous_beat_length = beats[i - 1][1]
            probability_Q7 = _lookup_probability(
                bigramBeatTransitions,
                bigramBeatTransitionProbabilities,
                previous_beat_length,
                beat_length,
            )

        probability_Q8 = _lookup_probability(
            bigramBeatPosTransitions,
            bigramBeatPosTransitionProbabilities,
            beat_position,
            beat_length,
        )

        log_probability_sum_Q7 += np.log(max(probability_Q7, EPS))
        log_probability_sum_Q8 += np.log(max(probability_Q8, EPS))

    perplexity_Q7 = np.exp(-log_probability_sum_Q7 / len(beats))
    perplexity_Q8 = np.exp(-log_probability_sum_Q8 / len(beats))

    return perplexity_Q7, perplexity_Q8


# Q9a
def beat_trigram_probability(midi_files):
    trigramBeatTransitions = defaultdict(list)
    trigramBeatTransitionProbabilities = defaultdict(list)

    trigram_counts = defaultdict(lambda: defaultdict(int))
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for i in range(1, len(beats)):
            previous_beat_length = beats[i - 1][1]
            beat_position, beat_length = beats[i]
            trigram_counts[(previous_beat_length, beat_position)][beat_length] += 1

    for key, length_counts in trigram_counts.items():
        total = sum(length_counts.values())
        for beat_length, count in length_counts.items():
            trigramBeatTransitions[key].append(beat_length)
            trigramBeatTransitionProbabilities[key].append(count / total)

    return trigramBeatTransitions, trigramBeatTransitionProbabilities


# Q9b
def beat_trigram_perplexity(midi_file):
    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    trigramBeatTransitions, trigramBeatTransitionProbabilities = beat_trigram_probability(midi_files)

    beats = beat_extraction(midi_file)
    if len(beats) == 0:
        return float('inf')

    log_probability_sum = 0.0
    for i, (beat_position, beat_length) in enumerate(beats):
        if i == 0:
            probability = _lookup_probability(
                bigramBeatPosTransitions,
                bigramBeatPosTransitionProbabilities,
                beat_position,
                beat_length,
            )
        else:
            previous_beat_length = beats[i - 1][1]
            probability = _lookup_probability(
                trigramBeatTransitions,
                trigramBeatTransitionProbabilities,
                (previous_beat_length, beat_position),
                beat_length,
            )
        log_probability_sum += np.log(max(probability, EPS))

    return np.exp(-log_probability_sum / len(beats))


# Q10
def music_generate(length):
    unigramProbabilities = note_unigram_probability(midi_files)
    bigramTransitions, bigramTransitionProbabilities = note_bigram_probability(midi_files)
    trigramTransitions, trigramTransitionProbabilities = note_trigram_probability(midi_files)

    sampled_notes = []
    if length > 0:
        sampled_notes.append(_sample_from_distribution(unigramProbabilities.keys(), unigramProbabilities.values()))
    if length > 1:
        first_note = sampled_notes[-1]
        if first_note in bigramTransitions:
            sampled_notes.append(_sample_from_distribution(bigramTransitions[first_note], bigramTransitionProbabilities[first_note]))
        else:
            sampled_notes.append(_sample_from_distribution(unigramProbabilities.keys(), unigramProbabilities.values()))

    while len(sampled_notes) < length:
        key = (sampled_notes[-2], sampled_notes[-1])
        if key in trigramTransitions:
            next_note = _sample_from_distribution(trigramTransitions[key], trigramTransitionProbabilities[key])
        elif sampled_notes[-1] in bigramTransitions:
            next_note = _sample_from_distribution(bigramTransitions[sampled_notes[-1]], bigramTransitionProbabilities[sampled_notes[-1]])
        else:
            next_note = _sample_from_distribution(unigramProbabilities.keys(), unigramProbabilities.values())
        sampled_notes.append(next_note)

    bigramBeatPosTransitions, bigramBeatPosTransitionProbabilities = beat_pos_bigram_probability(midi_files)
    beatUnigramProbabilities = _beat_length_unigram_probability(midi_files)

    sampled_beats = []
    beat_position = 0
    for _ in range(length):
        if beat_position in bigramBeatPosTransitions:
            beat_length = _sample_from_distribution(
                bigramBeatPosTransitions[beat_position],
                bigramBeatPosTransitionProbabilities[beat_position],
            )
        else:
            beat_length = _sample_from_distribution(beatUnigramProbabilities.keys(), beatUnigramProbabilities.values())
        sampled_beats.append(beat_length)
        beat_position = (beat_position + beat_length) % 32

    midi = MIDIFile(1)
    track = 0
    channel = 0
    tempo = 120
    volume = 100
    current_time = 0.0
    midi.addTempo(track, 0, tempo)

    for note, beat_length in zip(sampled_notes, sampled_beats):
        duration = beat_length / 8
        midi.addNote(track, channel, int(note), current_time, duration, volume)
        current_time += duration

    with open('q10.mid', 'wb') as f:
        midi.writeFile(f)
