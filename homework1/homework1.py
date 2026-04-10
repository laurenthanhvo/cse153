import glob
import math
import re

import numpy as np
from mido import MidiFile

SAMPLE_RATE = 44100


# Part A 

def note_name_to_frequency(note_name):
    match = re.fullmatch(r"([A-G]#?)(-?\d+)", note_name)
    if match is None:
        raise ValueError(f"Invalid note name: {note_name}")

    note, octave_str = match.groups()
    octave = int(octave_str)

    semitone_map = {
        "C": 0,
        "C#": 1,
        "D": 2,
        "D#": 3,
        "E": 4,
        "F": 5,
        "F#": 6,
        "G": 7,
        "G#": 8,
        "A": 9,
        "A#": 10,
        "B": 11,
    }

    midi_number = 12 * (octave + 1) + semitone_map[note]
    return 440.0 * (2.0 ** ((midi_number - 69) / 12.0))


def create_sine_wave(frequency, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)


def decrease_amplitude(audio):
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        return audio.copy()

    envelope = np.linspace(1.0, 0.0, num=len(audio), endpoint=True)
    return audio * envelope


def add_delay_effects(audio):
    audio = np.asarray(audio, dtype=float)
    delay_samples = int(0.5 * SAMPLE_RATE)

    delayed_audio = np.zeros(len(audio) + delay_samples, dtype=float)
    delayed_audio[: len(audio)] += 0.7 * audio
    delayed_audio[delay_samples : delay_samples + len(audio)] += 0.3 * audio
    return delayed_audio



def concatenate_audio(list_of_your_audio):
    if len(list_of_your_audio) == 0:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(audio, dtype=float) for audio in list_of_your_audio])



def mix_audio(list_of_your_audio, amplitudes):
    if len(list_of_your_audio) == 0:
        return np.array([], dtype=float)
    if len(list_of_your_audio) != len(amplitudes):
        raise ValueError("list_of_your_audio and amplitudes must have the same length")

    max_len = max(len(audio) for audio in list_of_your_audio)
    mixed = np.zeros(max_len, dtype=float)

    for audio, amp in zip(list_of_your_audio, amplitudes):
        audio = np.asarray(audio, dtype=float)
        mixed[: len(audio)] += amp * audio

    return mixed



def create_sawtooth_wave(frequency, duration, sample_rate=SAMPLE_RATE):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t, dtype=float)

    for k in range(1, 20):
        wave += ((-1) ** (k + 1) / k) * np.sin(2 * np.pi * k * frequency * t)

    wave *= 2.0 / np.pi
    return wave



def create_melody(notes, durations, sample_rate=SAMPLE_RATE, wave_type="sine"):
    waves = []
    for note, duration in zip(notes, durations):
        frequency = note_name_to_frequency(note) if isinstance(note, str) else float(note)
        if wave_type == "sine":
            wave = create_sine_wave(frequency, duration, sample_rate)
        elif wave_type == "sawtooth":
            wave = create_sawtooth_wave(frequency, duration, sample_rate)
        else:
            raise ValueError("wave_type must be 'sine' or 'sawtooth'")
        waves.append(wave)

    return concatenate_audio(waves)


# Part B 

def get_file_lists():
    piano_files = sorted(glob.glob("./piano/*.mid"))
    drum_files = sorted(glob.glob("./drums/*.mid"))
    return piano_files, drum_files



def get_num_beats(file_path):
    mid = MidiFile(file_path)
    max_track_ticks = 0

    for track in mid.tracks:
        cumulative_ticks = 0
        for msg in track:
            cumulative_ticks += msg.time
        max_track_ticks = max(max_track_ticks, cumulative_ticks)

    if mid.ticks_per_beat == 0:
        return 0.0
    return max_track_ticks / mid.ticks_per_beat



def get_stats(piano_path_list, drum_path_list):
    piano_beat_nums = [get_num_beats(file_path) for file_path in piano_path_list]
    drum_beat_nums = [get_num_beats(file_path) for file_path in drum_path_list]

    average_piano = float(np.mean(piano_beat_nums)) if piano_beat_nums else 0.0
    average_drum = float(np.mean(drum_beat_nums)) if drum_beat_nums else 0.0

    return {
        "piano_midi_num": len(piano_path_list),
        "drum_midi_num": len(drum_path_list),
        "average_piano_beat_num": average_piano,
        "average_drum_beat_num": average_drum,
    }



def _get_note_on_messages(file_path):
    mid = MidiFile(file_path)
    note_messages = []

    for track in mid.tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                note_messages.append(msg)

    return note_messages



def _get_note_numbers(file_path):
    return [msg.note for msg in _get_note_on_messages(file_path)]



def get_lowest_pitch(file_path):
    notes = _get_note_numbers(file_path)
    return min(notes) if notes else None



def get_highest_pitch(file_path):
    notes = _get_note_numbers(file_path)
    return max(notes) if notes else None



def get_unique_pitch_num(file_path):
    notes = _get_note_numbers(file_path)
    return len(set(notes))



def get_average_pitch_value(file_path):
    notes = _get_note_numbers(file_path)
    if not notes:
        return None
    return float(np.mean(notes))


def featureQ9(file_path):
    return [
        get_lowest_pitch(file_path),
        get_highest_pitch(file_path),
        get_unique_pitch_num(file_path),
        get_average_pitch_value(file_path),
    ]



def featureQ10(file_path):
    note_messages = _get_note_on_messages(file_path)
    notes = [msg.note for msg in note_messages]
    velocities = [msg.velocity for msg in note_messages]
    channels = [msg.channel for msg in note_messages if hasattr(msg, "channel")]

    if not notes:
        return [0.0] * 9

    unique_notes = set(notes)
    pitch_classes = {note % 12 for note in notes}

    lowest_pitch = min(notes) / 127.0
    highest_pitch = max(notes) / 127.0
    unique_pitch_fraction = len(unique_notes) / 128.0
    average_pitch = float(np.mean(notes)) / 127.0
    pitch_range = (max(notes) - min(notes)) / 127.0
    average_velocity = float(np.mean(velocities)) / 127.0
    pitch_class_fraction = len(pitch_classes) / 12.0

    n_beats = get_num_beats(file_path)
    note_density = len(notes) / n_beats if n_beats > 0 else 0.0
    normalized_density = min(note_density, 20.0) / 20.0

    if channels:
        channel9_fraction = sum(ch == 9 for ch in channels) / len(channels)
    else:
        channel9_fraction = 0.0

    return [
        lowest_pitch,
        highest_pitch,
        unique_pitch_fraction,
        average_pitch,
        pitch_range,
        average_velocity,
        pitch_class_fraction,
        normalized_density,
        channel9_fraction,
    ]