use crate::wave::{ChannelType, StereoWave, Wave, WaveForm, WaveMetaData};
use apodize::hanning_iter;
use phase_vocoder_helpers::split_stereo_wave;
use rust_interpolation::monointerp;
use rustfft::{algorithm::Radix4, num_complex::Complex, Fft, FftDirection};
use std::f32::consts::PI;
mod framed_vec;
use framed_vec::FramedVec;

pub struct PhaseVocoder {
    frame_size: usize,
    hop_size: usize,
    waveform: WaveForm,
    wave_metadata: WaveMetaData,
}

impl PhaseVocoder {
    pub fn new(wave: &Wave, frame_size: usize) -> Self {
        let hop_size = frame_size / 4; // hop size = N/4 for 75% frame overlap
        let padding_length = hop_size * 3;
        let retrieved_wave = wave.get_waveform();
        let waveform = phase_vocoder_helpers::pad_wave_with_zeros(retrieved_wave, padding_length);
        let wave_metadata = wave.get_metadata();

        Self {
            frame_size,
            hop_size,
            waveform,
            wave_metadata,
        }
    }

    // Shift function
    fn shift_channel(&self, waveform: &WaveForm, semitones: i16) -> WaveForm {
        // get the outward hop size, hop_out
        let two: f32 = 2.0;
        let alpha = two.powf((semitones as f32) / 12.0);
        let hop_out = (alpha * (self.hop_size as f32)).round() as usize;

        // Analysis
        println!("start analysis...");
        let frames = generate_frames(waveform.to_vec(), self.frame_size, self.hop_size);
        println!("number of samples = {}", frames.len());
        let fft_frames = fft(frames);
        println!("end analysis...");
        // End analysis

        // Processing
        println!("start processing...");
        let phase_differences = get_phase_difference(&fft_frames);
        let corrected_phase_differences = correct_phase_diffs(phase_differences, self.hop_size);
        let true_frequencies = get_true_frequency(corrected_phase_differences, self.hop_size);
        let final_phases = get_cumulative_phases(true_frequencies, hop_out);
        println!("end processing");
        // End processing

        // Synthesis
        println!("start synthesis...");
        let output_frames = inverse_fft(&fft_frames, &final_phases);
        let overlapped_waveform = overlap_add_frames(output_frames, hop_out, true);
        println!("end synthesis...");
        // End synthesis

        // Resampling
        println!("start resampling...");
        let output_waveform = resample(overlapped_waveform, alpha);
        println!("end resampling...");
        // End resampling

        return output_waveform;
    }

    pub fn shift_signal(&self, semitones: i16) -> Wave {
        // Handle Stereo signal

        let output_waveform: WaveForm = match self.wave_metadata.get_channels() {
            ChannelType::Mono => self.shift_channel(&self.waveform, semitones),
            // get as StereoWaveform, process each wave individually, then recombine into StereoWaveform
            // to get the final waveform
            ChannelType::Stereo => {
                let stereowave: StereoWave = split_stereo_wave(&self.waveform);
                let left = stereowave.get_left_channel();
                let right = stereowave.get_right_channel();
                let output_left = self.shift_channel(left, semitones);
                let output_right = self.shift_channel(right, semitones);
                let output_stereo = StereoWave::new(output_left, output_right).unwrap();
                let output: WaveForm = output_stereo.into();
                output
            }
        };

        Wave::new(
            output_waveform,
            self.wave_metadata.get_sample_rate(),
            self.wave_metadata.get_channels(),
            self.wave_metadata.get_bit_depth(),
            self.wave_metadata.get_data_length(),
            self.wave_metadata.get_audio_format(),
        )
    }
}

// Section 1: Analysis

fn generate_frames(samples: Vec<f32>, frame_size: usize, hop_size: usize) -> FramedVec<f32> {
    // frames now generated contiguously

    FramedVec::new(samples, frame_size, hop_size)
}

pub fn generate_hanning_window(length: usize) -> Vec<f32> {
    // generate a symmetrical Hanning window
    let window: Vec<f32> = hanning_iter(length * 2 + 1)
        .skip(1)
        .step_by(2)
        .map(|v| v as f32)
        .collect();
    return window;
}

pub fn apply_hanning_window(frame: &[f32], hop: usize) -> Vec<f32> {
    // normalisation of window compensates for overlap adding
    let normalisation = (((frame.len() / hop) / 2) as f32).sqrt();

    let window = generate_hanning_window(frame.len());

    let windowed_frame = frame
        .iter()
        .zip(window.iter())
        .map(|(data, w)| data * w / normalisation)
        .collect();

    return windowed_frame;
}

fn apply_fft_to_frame<T>(frame: &[T], direction: FftDirection) -> Vec<Complex<f32>>
where
    T: Into<Complex<f32>> + Copy,
{
    let fft_size = frame.len();
    let fft = Radix4::new(fft_size, direction);
    let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| x.into()).collect();
    fft.process(&mut buffer);
    return buffer;
}

fn fft(frames: FramedVec<f32>) -> FramedVec<Complex<f32>> {
    let mut fft_frames: FramedVec<Complex<f32>> =
        FramedVec::new_empty_non_overlapped(frames.len(), frames.frame_size());

    for frame in frames.iter() {
        let windowed_frame = apply_hanning_window(frame, frames.hop());
        let fft_frame = apply_fft_to_frame(&windowed_frame, FftDirection::Forward);

        fft_frames.push(fft_frame);
    }
    fft_frames.update_indices();

    // hop is now frame size because fft spectrum has no overlapping
    return fft_frames;
}

// End section 1

// Section 2: Processing

// we now have fixed frequency bins, but we want the true frequencies (which in general lie between bins)
// to do this we start by getting the phase differences between frames
fn get_phase_difference(fft_spectrum: &FramedVec<Complex<f32>>) -> FramedVec<f32> {
    let mut phase_diffs: Vec<f32> = Vec::with_capacity(fft_spectrum.len());

    // first frame needs to be done manually
    let first_frame_diffs: Vec<f32> = fft_spectrum
        .iter()
        .take(1)
        .flat_map(|frame| frame.iter().map(|c| c.arg()))
        .collect();
    for diff in first_frame_diffs {
        phase_diffs.push(diff);
    }

    // iterate over all consecutive frames and take the phase difference between them
    for (prev_frame, curr_frame) in fft_spectrum.iter().zip(fft_spectrum.iter().skip(1)) {
        let frame_diffs: Vec<f32> = prev_frame
            .iter()
            .zip(curr_frame.iter())
            .map(|(prev, curr)| {
                let prev_phase = prev.arg();
                let curr_phase = curr.arg();
                let phase_diff = curr_phase - prev_phase;

                phase_diff
            })
            .collect();

        for diff in frame_diffs {
            phase_diffs.push(diff);
        }
    }

    FramedVec::new(
        phase_diffs,
        fft_spectrum.frame_size(),
        fft_spectrum.frame_size(),
    )
}

fn correct_phase_diffs(phase_diffs: FramedVec<f32>, hop_size: usize) -> FramedVec<f32> {
    // We remove the expected phase difference due to natural accumulation
    // We also wrap back to [-π, π]
    let mut corrected_phase_diffs: Vec<f32> = Vec::with_capacity(phase_diffs.len());

    for diff in phase_diffs.iter() {
        let corrected_diff: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, &observed)| {
                let expected = (hop_size as f32) * 2.0 * PI * (i as f32) / (diff.len() as f32);
                phase_vocoder_helpers::modulo(observed - expected + PI, 2.0 * PI) - PI
            })
            .collect();
        for c_diff in corrected_diff {
            corrected_phase_diffs.push(c_diff);
        }
    }

    FramedVec::new(
        corrected_phase_diffs,
        phase_diffs.frame_size(),
        phase_diffs.frame_size(),
    )
}

// gets the true frequency bins from the given bins and the phase differences
fn get_true_frequency(phase_differences: FramedVec<f32>, hop_size: usize) -> FramedVec<f32> {
    let mut true_frequencies: Vec<f32> = Vec::with_capacity(phase_differences.len());
    for diff in phase_differences.iter() {
        let true_freq_frame: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, pd)| {
                let bin_freq = 2.0 * PI * (i as f32) / (diff.len() as f32);
                bin_freq + pd / (hop_size as f32)
            })
            .collect();
        for true_freq in true_freq_frame {
            true_frequencies.push(true_freq);
        }
    }

    FramedVec::new(
        true_frequencies,
        phase_differences.frame_size(),
        phase_differences.frame_size(),
    )
}

fn get_cumulative_phases(true_frequencies: FramedVec<f32>, hop_out: usize) -> FramedVec<f32> {
    let mut final_phases: FramedVec<f32> =
        FramedVec::new_empty_non_overlapped(true_frequencies.len(), true_frequencies.frame_size());

    for freq_frame in true_frequencies.iter() {
        let phase_diff_frame: Vec<f32> = freq_frame.iter().map(|f| f * (hop_out as f32)).collect();
        let default: Vec<f32> = vec![0.; phase_diff_frame.len()];
        // first frame no longer needs to be done manually
        let prev_frame = final_phases.iter().last().unwrap_or(&default);
        let cumulative_phase_frame =
            phase_vocoder_helpers::elementwise_add(prev_frame, &phase_diff_frame);
        final_phases.push(cumulative_phase_frame);
        // TODO I'm updating indices every iteration -- this can't be efficient! Try to optimise
        final_phases.update_indices();
    }
    return final_phases;
}

// End section 2

// Section 3: Synthesis

fn inverse_fft(fft_spectrum: &FramedVec<Complex<f32>>, phases: &FramedVec<f32>) -> FramedVec<f32> {
    let mut output_frames: FramedVec<f32> =
        FramedVec::new_empty_non_overlapped(fft_spectrum.len(), fft_spectrum.frame_size());
    for (fft_frame, phase_frame) in fft_spectrum.iter().zip(phases.iter()) {
        let rotated_frame = phase_vocoder_helpers::produce_output_frame(fft_frame, phase_frame);
        let output_frame: Vec<f32> = apply_fft_to_frame(&rotated_frame, FftDirection::Inverse)
            // take the real part and normalise
            .iter()
            .map(|c| c.re / (rotated_frame.len() as f32))
            .collect();
        output_frames.push(output_frame);
    }
    output_frames.update_indices();
    return output_frames;
}

// TODO I expect this function will need further optimising
fn overlap_add_frames(
    output_frames: FramedVec<f32>,
    hop_out: usize,
    apply_window: bool,
) -> Vec<f32> {
    let mut raw_frames: FramedVec<f32> =
        FramedVec::new_empty_non_overlapped(output_frames.len(), output_frames.frame_size());
    println!("frame size is {}", output_frames.frame_size());

    // apply window to each frame
    for frame in output_frames.iter() {
        let output_frame = if apply_window {
            apply_hanning_window(&frame, hop_out)
        } else {
            frame.to_vec()
        };
        raw_frames.push(output_frame);
    }
    raw_frames.update_indices();
    println!("raw frames length is {}", raw_frames.len());

    // set up the vectors
    let frame_length = raw_frames.frame_size();
    let new_waveform_size = (raw_frames.number_of_frames() - 1) * hop_out + frame_length;
    let mut overlapped_waveform: Vec<f32> = vec![0.; new_waveform_size];

    // first iteration manually
    let range = 0..frame_length;
    let default = vec![0.; frame_length];
    let mut frame_iterator = raw_frames.iter();
    let first_frame = frame_iterator.nth(0).unwrap_or(&default);
    let overlap_slice =
        phase_vocoder_helpers::elementwise_add(&overlapped_waveform[range.clone()], first_frame);
    let _ = overlapped_waveform.splice(range, overlap_slice);

    // overlap add by hop_out hop size
    for (i, frame) in frame_iterator.enumerate() {
        let time_index = (i + 1) * hop_out;
        let range = time_index..(time_index + frame_length);
        let overlap_slice =
            phase_vocoder_helpers::elementwise_add(&overlapped_waveform[range.clone()], frame);
        let _ = overlapped_waveform.splice(range, overlap_slice);
    }

    return overlapped_waveform;
}

// End section 3

// Section 4: Resampling

fn resample(waveform: WaveForm, alpha: f32) -> WaveForm {
    let x_axis: Vec<f32> = (0..waveform.len()).map(|v| v as f32).collect();
    let query: Vec<f32> = (0..)
        .map(|i| (i as f32) * alpha)
        .take_while(|&x| x < ((waveform.len() - 1) as f32))
        .collect();
    let resampled_waveform = monointerp(&query, &x_axis, &waveform).unwrap_or_default();
    return resampled_waveform;
}

// End section 4

mod phase_vocoder_helpers {
    use crate::wave::{StereoWave, WaveForm};
    use rustfft::num_complex::Complex;

    pub fn elementwise_add(v1: &[f32], v2: &[f32]) -> Vec<f32> {
        v1.into_iter().zip(v2).map(|(a, b)| a + b).collect()
    }

    pub fn modulo(a: f32, b: f32) -> f32 {
        ((a % b) + b) % b
    }

    pub fn produce_output_frame(
        fft_spectrum_frame: &[Complex<f32>],
        phase_frame: &[f32],
    ) -> Vec<Complex<f32>> {
        // The equation for this is X e^{i (2pi k n / N)}, summand for an inverse discrete FT
        // This is before performing inverse fft

        let rotated_spectrum_frame: Vec<Complex<f32>> = fft_spectrum_frame
            .iter()
            .zip(phase_frame.iter())
            .map(|(s, p)| s.norm() * Complex { re: 0.0, im: *p }.exp())
            .collect();

        return rotated_spectrum_frame;
    }

    pub fn pad_wave_with_zeros(waveform: &WaveForm, padding_length: usize) -> WaveForm {
        let mut padded_vec: WaveForm = vec![0.0; padding_length + waveform.len()];

        padded_vec[padding_length..].copy_from_slice(waveform);

        return padded_vec;
    }

    pub fn split_stereo_wave(samples: &Vec<f32>) -> StereoWave {
        let mut left_channel: Vec<f32> = Vec::with_capacity(samples.len() / 2);
        let mut right_channel: Vec<f32> = Vec::with_capacity(samples.len() / 2);
        for (i, sample) in samples.iter().enumerate() {
            if i % 2 == 0 {
                left_channel.push(*sample);
            } else {
                right_channel.push(*sample);
            }
        }
        // TODO deal with the potential error from this (atm it's just a channel length error)
        StereoWave::new(left_channel, right_channel).unwrap()
    }
}

#[cfg(test)]
mod tests {

    use super::*;
    use phase_vocoder_helpers::*;

    #[test]
    fn test_elementwise_add() {
        let v1: Vec<f32> = vec![1.0, 2.0, 3.0];
        let v2: Vec<f32> = vec![10.0, 11.0, 12.0];
        let out_vec: Vec<f32> = vec![11.0, 13.0, 15.0];
        assert_eq!(elementwise_add(&v1, &v2), out_vec);
    }

    // Test Section 1: Analysis

    #[test]
    fn test_generate_frames() {
        let input: Vec<f32> = vec![0., 1., 2., 3., 4., 5.];
        let output = vec![
            vec![0., 1., 2.],
            vec![1., 2., 3.],
            vec![2., 3., 4.],
            vec![3., 4., 5.],
            vec![4., 5., 0.],
            vec![5., 0., 0.],
        ];
        let frames = generate_frames(input, 3, 1);
        for (i, frame) in frames.iter().enumerate() {
            assert_eq!(frame, output[i]);
        }
    }

    #[test]
    fn hanning_window_generation() {
        let hanning_test: Vec<f32> = vec![
            0.0, 0.11697778, 0.41317591, 0.75, 0.96984631, 0.96984631, 0.75, 0.41317591,
            0.11697778, 0.0,
        ];
        let hanning_generated = generate_hanning_window(10);
        for (v1, v2) in hanning_test.iter().zip(hanning_generated.iter()) {
            assert!(v1 - v2 < 0.0001);
        }
    }

    #[test]
    fn fft_test() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let fft_vec: Vec<f32> = apply_fft_to_frame(&v, FftDirection::Forward)
            .iter()
            .map(|v| v.re)
            .collect();
        println!("{:?}", fft_vec);
        let out_v: Vec<f32> = vec![
            24.000000, -6.828427, 0.000000, -1.171573, 0.000000, -1.171573, 0.000000, -6.828427,
        ];
        for (ff, vv) in fft_vec.iter().zip(out_v.iter()) {
            assert!((ff - vv).abs() < 0.0001);
        }
    }

    // End Test Section 1

    // Test Section 2: Processing

    #[test]
    fn phase_difference_test() {
        let my_vec = vec![
            Complex { re: 1.0, im: 0.0 },
            Complex { re: 1.0, im: 1.0 },
            Complex { re: 0.0, im: 1.0 },
            Complex { re: 0.0, im: 1.0 },
        ];

        let example_fft: FramedVec<Complex<f32>> = FramedVec::new(my_vec, 2, 2);
        let output_vec = vec![0.0, PI / 4.0, PI / 2.0, PI / 4.0];
        let output = FramedVec::new(output_vec, 2, 2);
        let phase_diffs = get_phase_difference(&example_fft);
        assert_eq!(output, phase_diffs);
    }

    // c = s - 2πi/N
    #[test]
    fn test_correct_phase_diffs() {
        let my_vec = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let input = FramedVec::new(my_vec.clone(), 3, 3);
        let corrected_pds = correct_phase_diffs(input, 1).clone_vec();
        let output_vec: Vec<f32> = my_vec
            .iter()
            .enumerate()
            .map(|(i, v)| {
                phase_vocoder_helpers::modulo(v - 2. * PI * (i as f32) / 3. + PI, 2. * PI) - PI
            })
            .collect();
        for (x, y) in corrected_pds.iter().zip(output_vec.iter()) {
            println!("x: {x}, y: {y}");
            assert!((x - y).abs() < 0.0001)
        }
    }

    #[test]
    fn test_true_frequency() {
        let input: Vec<f32> = vec![0., 1., 2., 3., 4., 5.];
        let phase_differences = FramedVec::new(input.clone(), 3, 3);
        let true_freq = get_true_frequency(phase_differences, 2).clone_vec();
        let output_vec: Vec<f32> = input
            .iter()
            .enumerate()
            .map(|(i, v)| 2. * PI * modulo(i as f32, 3.) / 3. + v / 2.)
            .collect();
        for (x, y) in true_freq.iter().zip(output_vec.iter()) {
            println!("x: {x}, y: {y}");
            assert!((x - y).abs() < 0.0001);
        }
    }

    #[test]
    fn modulo_test() {
        let x: f32 = phase_vocoder_helpers::modulo(3.0 * PI, 2.0 * PI);
        assert!((x - PI).abs() < 0.0001);
    }

    #[test]
    fn test_cumulative_phases() {
        let input: Vec<f32> = vec![0., 1., 2., 3., 4., 5.];
        let true_freqs = FramedVec::new(input, 3, 3);
        let cumulative_phases = get_cumulative_phases(true_freqs, 1).clone_vec();
        let output_vec: Vec<f32> = vec![0., 1., 2., 3., 5., 7.];
        for (x, y) in cumulative_phases.iter().zip(output_vec.iter()) {
            println!("x: {x}, y: {y}");
            assert!((x - y).abs() < 0.0001);
        }
    }

    // End Test Section 2

    // Test Section 3: Synthesis

    #[test]
    fn test_inverse_fft() {
        let f: Vec<Complex<f32>> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0]
            .iter()
            .map(|n| n.into())
            .collect();
        let p: Vec<f32> = vec![PI; f.len()];
        let freqs = FramedVec::new(f, 8, 8);
        let phases = FramedVec::new(p, 8, 8);
        let ifft_vec = inverse_fft(&freqs, &phases).clone_vec();
        let output: Vec<f32> = vec![-3., 0.853553, 0., 0.146447, 0., 0.146447, 0., 0.853553];
        for (x, y) in ifft_vec.iter().zip(output.iter()) {
            println!("x: {x}, y: {y}");
            assert!((x - y).abs() < 0.0001);
        }
    }

    #[test]
    fn test_output_frame() {
        let fft_sim: Vec<Complex<f32>> = vec![
            Complex::from(1.0),
            Complex::from(1.0),
            Complex::from(2.0_f32.sqrt()),
        ];
        let phase_sim: Vec<f32> = vec![PI, PI / 2.0, PI / 4.0];
        let output = phase_vocoder_helpers::produce_output_frame(&fft_sim, &phase_sim);
        let compare: Vec<Complex<f32>> = vec![
            Complex::from(-1.0),
            Complex { re: 0.0, im: 1.0 },
            Complex { re: 1.0, im: 1.0 },
        ];
        for (o, c) in output.iter().zip(compare.iter()) {
            assert!((o - c).re.abs() < 0.0001);
        }
    }

    #[test]
    fn test_overlap_add() {
        let hop = 1;
        let my_vec = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input: FramedVec<f32> = FramedVec::new(my_vec, 3, 3);
        let output: Vec<f32> = vec![1.0, 6.0, 8.0, 6.0];
        assert_eq!(output, overlap_add_frames(input, hop, false));
    }

    // End Test Section 3

    // Test Section 4 Resampling

    // End Test Section 4

    // Overall Tests
}
