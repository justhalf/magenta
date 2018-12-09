# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Create the recordio files necessary for training onsets and frames.

The training files are split in ~20 second chunks by default, the test files
are not split.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bisect
import glob
import math
import os
import re
import sys
import copy

import librosa
import numpy as np
import tensorflow as tf

from magenta.music import audio_io
from magenta.music import midi_io
from magenta.music import sequences_lib
from magenta.protobuf import music_pb2
from mir_eval.transcription import match_notes
import pretty_midi
from magenta.music.midi_io import midi_to_sequence_proto
from magenta.models.onsets_frames_transcription.infer_util import sequence_to_valued_intervals_notes

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input_dir', None,
                           'Directory where the un-zipped MAPS files are.')
tf.app.flags.DEFINE_string('predicted_dir', None,
                           'Directory where the predicted midis and wavs are (for test set).')
tf.app.flags.DEFINE_string('output_dir', './',
                           'Directory where the two output TFRecord files '
                           '(train and test) will be placed.')
tf.app.flags.DEFINE_integer('min_length', 5, 'minimum segment length')
tf.app.flags.DEFINE_integer('max_length', 20, 'maximum segment length')
tf.app.flags.DEFINE_integer('sample_rate', 16000, 'desired sample rate')
tf.app.flags.DEFINE_string('mode', 'standard',
                           'Whether to create standard dataset or resynth dataset')
tf.app.flags.DEFINE_string('resynth_mid_suffix', '_removed',
                           'The suffix to be found at the end of training resynth midi files')
tf.app.flags.DEFINE_string('resynth_wav_suffix', '_removed_125',
                           'The suffix to be found at the end of training resynth WAV files') #resynth_wav_suffix = nama flag, _removed_125 = default value


test_dirs = ['ENSTDkCl/MUS', 'ENSTDkAm/MUS']
# test_dirs = ['ENSTDkCl/MUS']
train_dirs = ['AkPnBcht/MUS', 'AkPnBsdf/MUS', 'AkPnCGdD/MUS', 'AkPnStgb/MUS',
	      'SptkBGAm/MUS', 'SptkBGCl/MUS', 'StbgTGd2/MUS']
# train_dirs = ['AkPnBcht/MUS']
test_resynth_dirs = ['ENSTDkCl', 'ENSTDkAm']
train_resynth_dirs = ['AkPnBcht', 'AkPnBsdf', 'AkPnCGdD', 'AkPnStgb',
	   	      'SptkBGAm', 'SptkBGCl', 'StbgTGd2']
MAPS_DIRS = set(train_resynth_dirs + test_resynth_dirs)

def get_MAPS_dirname(filename):
    for token in filename.split('_')[2:]:
      if token in MAPS_DIRS:
        return token
    raise ValueError('The given filename {} does not contain MAPS dirname'.format(filename))

def _find_inactive_ranges(note_sequence):
  """Returns ranges where no notes are active in the note_sequence."""
  start_sequence = sorted(
      note_sequence.notes, key=lambda note: note.start_time, reverse=True)
  end_sequence = sorted(
      note_sequence.notes, key=lambda note: note.end_time, reverse=True)

  notes_active = 0

  time = start_sequence[-1].start_time
  inactive_ranges = []
  if time > 0:
    inactive_ranges.append(0.)
    inactive_ranges.append(time)
  start_sequence.pop()
  notes_active += 1
  # Iterate through all note on events
  while start_sequence or end_sequence:
    if start_sequence and (start_sequence[-1].start_time <
                           end_sequence[-1].end_time):
      if notes_active == 0:
        time = start_sequence[-1].start_time
        inactive_ranges.append(time)
      notes_active += 1
      start_sequence.pop()
    else:
      notes_active -= 1
      if notes_active == 0:
        time = end_sequence[-1].end_time
        inactive_ranges.append(time)
      end_sequence.pop()

  # if the last note is the same time as the end, don't add it
  # remove the start instead of creating a sequence with 0 length
  if inactive_ranges[-1] < note_sequence.total_time:
    inactive_ranges.append(note_sequence.total_time)
  else:
    inactive_ranges.pop()

  assert len(inactive_ranges) % 2 == 0

  inactive_ranges = [(inactive_ranges[2 * i], inactive_ranges[2 * i + 1])
                     for i in range(len(inactive_ranges) // 2)]
  return inactive_ranges


def _last_zero_crossing(samples, start, end):
  """Returns the last zero crossing in the window [start, end)."""
  samples_greater_than_zero = samples[start:end] > 0
  samples_less_than_zero = samples[start:end] < 0
  samples_greater_than_equal_zero = samples[start:end] >= 0
  samples_less_than_equal_zero = samples[start:end] <= 0

  # use np instead of python for loop for speed
  xings = np.logical_or(
      np.logical_and(samples_greater_than_zero[:-1],
                     samples_less_than_equal_zero[1:]),
      np.logical_and(samples_less_than_zero[:-1],
                     samples_greater_than_equal_zero[1:])).nonzero()[0]

  return xings[-1] + start if xings.size > 0 else None


def find_split_points(note_sequence, samples, sample_rate, min_length,
                      max_length):
  """Returns times at which there are no notes.

  The general strategy employed is to first check if there are places in the
  sustained pianoroll where no notes are active within the max_length window;
  if so the middle of the last gap is chosen as the split point.

  If not, then it checks if there are places in the pianoroll without sustain
  where no notes are active and then finds last zero crossing of the wav file
  and chooses that as the split point.

  If neither of those is true, then it chooses the last zero crossing within
  the max_length window as the split point.

  If there are no zero crossings in the entire window, then it basically gives
  up and advances time forward by max_length.

  Args:
      note_sequence: The NoteSequence to split.
      samples: The audio file as samples.
      sample_rate: The sample rate (samples/second) of the audio file.
      min_length: Minimum number of seconds in a split.
      max_length: Maximum number of seconds in a split.

  Returns:
      A list of split points in seconds from the beginning of the file.
  """

  if not note_sequence.notes:
    return []

  end_time = note_sequence.total_time

  note_sequence_sustain = sequences_lib.apply_sustain_control_changes(
      note_sequence)

  ranges_nosustain = _find_inactive_ranges(note_sequence)
  ranges_sustain = _find_inactive_ranges(note_sequence_sustain)

  nosustain_starts = [x[0] for x in ranges_nosustain]
  sustain_starts = [x[0] for x in ranges_sustain]

  nosustain_ends = [x[1] for x in ranges_nosustain]
  sustain_ends = [x[1] for x in ranges_sustain]

  split_points = [0.]

  while end_time - split_points[-1] > max_length:
    max_advance = split_points[-1] + max_length

    # check for interval in sustained sequence
    pos = bisect.bisect_right(sustain_ends, max_advance)
    if pos < len(sustain_starts) and max_advance > sustain_starts[pos]:
      split_points.append(max_advance)

    # if no interval, or we didn't fit, try the unmodified sequence
    elif pos == 0 or sustain_starts[pos - 1] <= split_points[-1] + min_length:
      # no splits available, use non sustain notes and find close zero crossing
      pos = bisect.bisect_right(nosustain_ends, max_advance)

      if pos < len(nosustain_starts) and max_advance > nosustain_starts[pos]:
        # we fit, great, try to split at a zero crossing
        zxc_start = nosustain_starts[pos]
        zxc_end = max_advance
        last_zero_xing = _last_zero_crossing(
            samples,
            int(math.floor(zxc_start * sample_rate)),
            int(math.ceil(zxc_end * sample_rate)))
        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and just return where there are at least no notes
          split_points.append(max_advance)

      else:
        # there are no good places to cut, so just pick the last zero crossing
        # check the entire valid range for zero crossings
        start_sample = int(
            math.ceil((split_points[-1] + min_length) * sample_rate)) + 1
        end_sample = start_sample + (max_length - min_length) * sample_rate
        last_zero_xing = _last_zero_crossing(samples, start_sample, end_sample)

        if last_zero_xing:
          last_zero_xing = float(last_zero_xing) / sample_rate
          split_points.append(last_zero_xing)
        else:
          # give up and advance by max amount
          split_points.append(max_advance)
    else:
      # only advance as far as max_length
      new_time = min(np.mean(ranges_sustain[pos - 1]), max_advance)
      split_points.append(new_time)

  if split_points[-1] != end_time:
    split_points.append(end_time)

  # ensure that we've generated a valid sequence of splits
  for prev, curr in zip(split_points[:-1], split_points[1:]):
    assert curr > prev
    assert curr - prev <= max_length + 1e-8
    if curr < end_time:
      assert curr - prev >= min_length - 1e-8
  assert end_time - split_points[-1] < max_length

  return split_points


def filename_to_id(filename):
  """Translate a .wav or .mid path to a MAPS sequence id."""
  return re.match(r'.*MUS-(.*)_[^_]+\.\w{3}',
                  os.path.basename(filename)).group(1)


def generate_train_set(exclude_ids):
  """Generate the train TFRecord."""
  train_file_pairs = []
  for directory in train_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      if filename_to_id(wav_file) not in exclude_ids:
        train_file_pairs.append((wav_file, mid_file))

  train_output_name = os.path.join(FLAGS.output_dir,
                                   'maps_config2_train.tfrecord')

  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for pair in train_file_pairs:
      print(pair)
      # load the wav data
      wav_data = tf.gfile.Open(pair[0], 'rb').read()
      samples = audio_io.wav_data_to_samples(wav_data, FLAGS.sample_rate)
      samples = librosa.util.normalize(samples, norm=np.inf)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      splits = find_split_points(ns, samples, FLAGS.sample_rate,
                                 FLAGS.min_length, FLAGS.max_length)

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      for start, end in zip(splits[:-1], splits[1:]):
        if end - start < FLAGS.min_length:
          continue

        new_ns = sequences_lib.extract_subsequence(ns, start, end)
        new_wav_data = audio_io.crop_wav_data(wav_data, FLAGS.sample_rate,
                                              start, end - start)
        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[pair[0]]
                )),
            'sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_ns.SerializeToString()]
                )),
            'audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            }))
        writer.write(example.SerializeToString())


def generate_test_set():
  """Generate the test TFRecord."""
  test_file_pairs = []
  for directory in test_dirs:
    path = os.path.join(FLAGS.input_dir, directory)
    path = os.path.join(path, '*.wav')
    wav_files = glob.glob(path)
    # find matching mid files
    for wav_file in wav_files:
      base_name_root, _ = os.path.splitext(wav_file)
      mid_file = base_name_root + '.mid'
      test_file_pairs.append((wav_file, mid_file))

  test_output_name = os.path.join(FLAGS.output_dir,
                                  'maps_config2_test.tfrecord')

  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for pair in test_file_pairs:
      print(pair)
      # load the wav data and resample it.
      samples = audio_io.load_audio(pair[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      ns = midi_io.midi_file_to_note_sequence(pair[1])

      velocities = [note.velocity for note in ns.notes]
      velocity_max = np.max(velocities)
      velocity_min = np.min(velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      example = tf.train.Example(features=tf.train.Features(feature={
          'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[pair[0]]
              )),
          'sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[ns.SerializeToString()]
              )),
          'audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[wav_data]
              )),
          'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[new_velocity_tuple.SerializeToString()]
              )),
          }))
      writer.write(example.SerializeToString())


  print(test_file_pairs)
  return [filename_to_id(wav) for wav, _ in test_file_pairs]


def generate_train_set_resynth(exclude_ids):
  """Generate the train TFRecord."""
  train_file_quads = []
  # For training there are four files:
  # MAPS_MUS-ty_mai_SptkBGAm.mid
  # MAPS_MUS-ty_mai_SptkBGAm_removed.mid
  # MAPS_MUS-ty_mai_SptkBGAm.wav
  # MAPS_MUS-ty_mai_SptkBGAm_removed_125.wav
  path = os.path.join(FLAGS.input_dir, '*{}.mid'.format(FLAGS.resynth_mid_suffix))
  #print("path: " + str(path))
  mid_files = glob.glob(path)
  # find matching wav files
  #print("----------- here ------------")
  #print(mid_files)
  for resynth_mid_file in mid_files:
    # base_name_root is absolute path

    base_name_root, _ = os.path.splitext(resynth_mid_file) #base_name_root should be MAPS_MUS-alb_se3_AkPnBcht
    base_name_root_list = base_name_root.split('_')
    base_name_root = "_".join(base_name_root_list[:-1])

    #print("------ BASE NAME ROOT: ---- " + str(base_name_root))
    orig_mid_file = '{}.mid'.format(base_name_root)
    #print("========= orig_mid_file: " + str(orig_mid_file))

    resynth_wav_file = '{}{}.wav'.format(base_name_root, FLAGS.resynth_wav_suffix)
    #print("========= resynth_wav_file: " + str(resynth_wav_file))
    
    orig_wav_file = '{}.wav'.format(base_name_root)
    #print("========= orig_wav_file: " + str(orig_wav_file))
    #print(exclude_ids)
    MAPS_dirname = get_MAPS_dirname(resynth_mid_file)
    # if MAPS_dirname not in exclude_ids:
    #   train_file_quads.append((orig_wav_file, resynth_wav_file,
		#                      orig_mid_file, resynth_mid_file))
    if MAPS_dirname in train_resynth_dirs:
      train_file_quads.append((orig_wav_file, resynth_wav_file,
                         orig_mid_file, resynth_mid_file))

  train_output_name = os.path.join(FLAGS.output_dir,
                                   'maps_config2_train_resynth.tfrecord')
  #print("train output name: " + str(train_output_name))
  with tf.python_io.TFRecordWriter(train_output_name) as writer:
    for quad in train_file_quads:
      print("quad: " + str(quad))
      # load the orig wav data
      orig_wav_data = tf.gfile.Open(quad[0], 'rb').read()
      orig_samples = audio_io.wav_data_to_samples(orig_wav_data, FLAGS.sample_rate)
      orig_samples = librosa.util.normalize(orig_samples, norm=np.inf)
      # load the resynth wav data
      resynth_wav_data = tf.gfile.Open(quad[1], 'rb').read()
      resynth_samples = audio_io.wav_data_to_samples(resynth_wav_data, FLAGS.sample_rate)
      resynth_samples = librosa.util.normalize(resynth_samples, norm=np.inf)

      # load the midi data and convert to a notesequence
      orig_ns = midi_io.midi_file_to_note_sequence(quad[2])
      resynth_ns = midi_io.midi_file_to_note_sequence(quad[3])
      splits = find_split_points(orig_ns, orig_samples, FLAGS.sample_rate,
                                 FLAGS.min_length, FLAGS.max_length)

      orig_velocities = [note.velocity for note in orig_ns.notes]
      velocity_max = np.max(orig_velocities)
      velocity_min = np.min(orig_velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)
      print("down here")
      for start, end in zip(splits[:-1], splits[1:]):
        print("inner for")
        sys.stdout.flush()
        if end - start < FLAGS.min_length:
          continue

        new_orig_ns = sequences_lib.extract_subsequence(orig_ns, start, end)
        new_orig_wav_data = audio_io.crop_wav_data(orig_wav_data, FLAGS.sample_rate,
                                              start, end - start)
        new_resynth_ns = sequences_lib.extract_subsequence(resynth_ns, start, end)
        new_resynth_wav_data = audio_io.crop_wav_data(resynth_wav_data, FLAGS.sample_rate,
                                              start, end - start)
        new_diff_ns = difference_note_sequence(new_orig_ns, new_resynth_ns)
        example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[quad[0]]
                )),
            'orig_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_orig_ns.SerializeToString()]
                )),
            'resynth_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_resynth_ns.SerializeToString()]
                )),
            'orig_audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_orig_wav_data]
                )),
            'resynth_audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_resynth_wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            'diff_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_diff_ns.SerializeToString()]
                )),
            }))
        writer.write(example.SerializeToString())


def _parse(example_proto):
  features = {
      'id': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'orig_sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'resynth_sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'orig_audio': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'resynth_audio': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'velocity_range': tf.FixedLenFeature(shape=(), dtype=tf.string),
      'diff_sequence': tf.FixedLenFeature(shape=(), dtype=tf.string),
  }
  return tf.parse_single_example(example_proto, features)


def fix_train_set_resynth(input_name, output_name):
  """Generate the train TFRecord."""
  if not os.path.exists(input_name):
    print('Path {} does not exist'.format(input_name))
    sys.exit(1)
  #print("train output name: " + str(train_output_name))
  def get_val(record, field):
    return record.features.feature[field].bytes_list.value[0]
  prev_path = ''
  counter = 0
  with tf.python_io.TFRecordWriter(output_name) as writer:
    for string_record in tf.python_io.tf_record_iterator(input_name):
      record = tf.train.Example()
      record.ParseFromString(string_record)
      path = get_val(record, 'id')
      if path != prev_path:
        counter = 0
        print(path)
        prev_path = path
      counter += 1
      # print('part {}'.format(counter))
      orig_ns = music_pb2.NoteSequence.FromString(get_val(record, 'orig_sequence'))
      resynth_ns = music_pb2.NoteSequence.FromString(get_val(record, 'resynth_sequence'))
      # diff_ns = music_pb2.NoteSequence.FromString(get_val(record, 'diff_sequence'))
      try:
          new_diff_ns = difference_note_sequence(orig_ns, resynth_ns)
      except ValueError:
          continue
      example = tf.train.Example(features=tf.train.Features(feature={
          'id':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[path]
              )),
          'orig_sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[orig_ns.SerializeToString()]
              )),
          'resynth_sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[resynth_ns.SerializeToString()]
              )),
          'orig_audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[get_val(record, 'orig_audio')]
              )),
          'resynth_audio':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[get_val(record, 'resynth_audio')]
              )),
          'velocity_range':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[get_val(record, 'velocity_range')]
              )),
          'diff_sequence':
          tf.train.Feature(bytes_list=tf.train.BytesList(
              value=[new_diff_ns.SerializeToString()]
              )),
          }))
      writer.write(example.SerializeToString())


def get_base_name_from_super_long_filename(super_long_filename):
  #super_long_filename = 'data/MAPS_predicted/_usr2_home_amuis_.other_speech_data_MAPS_ENSTDkAm_MUS_MAPS_MUS-bk_xmas1_ENSTDkAm.wav_label_from_frames.mid'
  name, ext = os.path.splitext(super_long_filename)
  start = -1
  end = -1
  tokens = re.split('[._]', name)
  for idx, token in enumerate(tokens):
    if token == 'MAPS':
      start = idx
    if token in MAPS_DIRS:
      end = idx+1
  base_name = '_'.join(tokens[start:end])
  return base_name

def generate_test_set_resynth():
  """Generate the test TFRecord for resynth dataset."""
  test_file_quadruples = []

  #--------
  
  #print("here")
  #print(FLAGS.predicted_dir)
  #print(FLAGS.input_dir)
  path = os.path.join(FLAGS.predicted_dir, '*.wav.mid')
  mid_files = glob.glob(path)
  # find matching wav files
  #print("----------- TEST here ------------")
  #print(mid_files)
  
  for predicted_mid_file in mid_files:
    # base_name_root is absolute path

    base_name_root = get_base_name_from_super_long_filename(predicted_mid_file) #base_name_root should be MAPS_MUS-alb_se3_AkPnBcht
    #print("------ BASE NAME ROOT: ---- " + str(base_name_root))

    orig_mid_file = '{}/{}.mid'.format(FLAGS.input_dir, base_name_root)
    #print("orig_mid_file: " + str(orig_mid_file))

    predicted_wav_file = '{}.wav'.format(predicted_mid_file[:len(predicted_mid_file)-4])
    #print("predicted_wav_file: " + str(predicted_wav_file))
    
    orig_wav_file = '{}/{}.wav'.format(FLAGS.input_dir, base_name_root)
    #print("orig_wav_file: " + str(orig_wav_file))

    test_file_quadruples.append((orig_wav_file, predicted_wav_file,
                         orig_mid_file, predicted_mid_file))
  #--------

  test_output_name = os.path.join(FLAGS.output_dir,
                                  'maps_config2_test_resynth.tfrecord')

  with tf.python_io.TFRecordWriter(test_output_name) as writer:
    for quadruple in test_file_quadruples:
      print(quadruple)
      # load the wav data and resample it.
      samples = audio_io.load_audio(quadruple[0], FLAGS.sample_rate)
      wav_data = audio_io.samples_to_wav_data(samples, FLAGS.sample_rate)
      samples_resynth = audio_io.load_audio(quadruple[1], FLAGS.sample_rate) #predicted
      predicted_wav_data = audio_io.samples_to_wav_data(samples_resynth, FLAGS.sample_rate)

      # load the midi data and convert to a notesequence
      orig_ns = midi_io.midi_file_to_note_sequence(quadruple[2])
      predicted_ns = midi_io.midi_file_to_note_sequence(quadruple[3])
      diff_ns = difference_note_sequence(orig_ns, predicted_ns)

      #import pdb
      #pdb.set_trace()
      #print(orig_ns.instrument[0])

      orig_velocities = [note.velocity for note in orig_ns.notes]
      velocity_max = np.max(orig_velocities)
      velocity_min = np.min(orig_velocities)
      new_velocity_tuple = music_pb2.VelocityRange(
          min=velocity_min, max=velocity_max)

      example = tf.train.Example(features=tf.train.Features(feature={
            'id':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[quadruple[0]]
                )),
            'orig_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[orig_ns.SerializeToString()]
                )),
            'resynth_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[predicted_ns.SerializeToString()]
                )),
            'orig_audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[wav_data]
                )),
            'resynth_audio':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[predicted_wav_data]
                )),
            'velocity_range':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[new_velocity_tuple.SerializeToString()]
                )),
            'diff_sequence':
            tf.train.Feature(bytes_list=tf.train.BytesList(
                value=[diff_ns.SerializeToString()]
                )),
            }))
      writer.write(example.SerializeToString())

  return [filename_to_id(wav) for wav, _, _, _ in test_file_quadruples]

def difference_note_sequence(gold_ns, silver_ns):
  ref_intervals, ref_pitches, ref_notes = sequence_to_valued_intervals_notes(gold_ns, 0)
  pred_intervals, pred_pitches, pred_notes = sequence_to_valued_intervals_notes(silver_ns, 0)
  #https://craffel.github.io/mir_eval/#mir_eval.transcription.match_notes
  # paired_idx is a list of tuple (ref_idx, pred_idx)
  try:
      paired_idx = match_notes(ref_intervals, pretty_midi.note_number_to_hz(ref_pitches),
                               pred_intervals, pretty_midi.note_number_to_hz(pred_pitches),
                               offset_ratio=None)
  except:
      raise ValueError('Empty note sequence')

  ref_idx = range(len(ref_intervals))
  not_missed_ref_idx = [i[0] for i in paired_idx]

  missing_notes = [ref_notes[element] for element in ref_idx if element not in not_missed_ref_idx]

  sequence = copy.deepcopy(gold_ns)
  del gold_ns.notes[:]
  for missing_note in missing_notes:
    note = sequence.notes.add()
    note.start_time = missing_note.start_time
    note.end_time = missing_note.end_time
    note.pitch = missing_note.pitch
    note.velocity = missing_note.velocity
    note.instrument = missing_note.instrument
    note.program = missing_note.program

  return sequence

def main(unused_argv):
  if FLAGS.mode == 'standard':
    test_ids = generate_test_set()
    generate_train_set(test_ids)
  else:
    '''
    silver_midi = "data/MAPS_predicted/_usr2_home_amuis_.other_speech_data_MAPS_ENSTDkAm_MUS_MAPS_MUS-bk_xmas1_ENSTDkAm.wav_label_from_frames.mid"
    gold_midi = "data/MAPS_resynth/MAPS_MUS-bk_xmas1_ENSTDkAm.mid"
    difference_note_sequence(gold_midi, silver_midi)
    
    #result = get_base_name_from_super_long_filename(silver_midi)
    #print(result)
    '''
    # test_ids = generate_test_set_resynth()
    # print(test_ids)
    # print("finish testing")
    # generate_train_set_resynth(test_ids)
    print('Fixing {}/maps_config2_test_resynth.tfrecord'.format(FLAGS.output_dir))
    test_input_name = os.path.join(FLAGS.output_dir,
                                     'maps_config2_test_resynth_old.tfrecord')
    test_output_name = os.path.join(FLAGS.output_dir,
                                     'maps_config2_test_resynth.tfrecord')
    fix_train_set_resynth(test_input_name, test_output_name)
    print('Fixing {}/maps_config2_train_resynth.tfrecord'.format(FLAGS.output_dir))
    train_input_name = os.path.join(FLAGS.output_dir,
                                     'maps_config2_train_resynth_old.tfrecord')
    train_output_name = os.path.join(FLAGS.output_dir,
                                     'maps_config2_train_resynth.tfrecord')
    fix_train_set_resynth(train_input_name, train_output_name)
    print("finish training")
    


def console_entry_point():
  tf.app.run(main)


if __name__ == '__main__':
  console_entry_point()
