use crate::wave::{ChannelType, StereoWave, Wave, WaveForm, WaveMetaData};
use phase_vocoder_helpers::split_stereo_wave;
use rust_interpolation::monointerp;
use rustfft::{algorithm::Radix4, num_complex::Complex, Fft, FftDirection};
use std::f32::consts::PI;

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
        let frames = generate_frames(&waveform, self.frame_size, self.hop_size);
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
                let output_left = self.shift_channel(&left, semitones);
                let output_right = self.shift_channel(&right, semitones);
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

fn generate_frames(samples: &Vec<f32>, frame_size: usize, hop_size: usize) -> Vec<Vec<f32>> {
    // calculate number of frames
    let n_frames = 1 + (samples.len() - frame_size) / hop_size; // this line may need redoing for safety, e.g. ensuring n_frames is positive

    // create an empty vec for the frames
    // TODO optimise by using Vec<[T; N]>, or ndarray crate, or Vec<Box<[T]>>
    let mut frames: Vec<Vec<f32>> = Vec::with_capacity(n_frames);

    // iterate over frames, multiplying waveform by hanning window and populating Vec
    for i in 0..n_frames {
        let start = i * hop_size;
        let end = start + frame_size;
        let new_frame = phase_vocoder_helpers::apply_hanning_window(&samples[start..end], hop_size);
        frames.push(new_frame);
    }

    let n_frames = frames.len();
    println!("number of frames is {n_frames}");
    println!("number of samples is {}", samples.len());

    return frames;
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

fn fft(frames: Vec<Vec<f32>>) -> Vec<Vec<Complex<f32>>> {
    let mut fft_frames: Vec<Vec<Complex<f32>>> = Vec::with_capacity(frames.len());
    for frame in frames {
        let fft_frame = apply_fft_to_frame(&frame, FftDirection::Forward);
        fft_frames.push(fft_frame);
    }
    return fft_frames;
}

// End section 1

// Section 2: Processing

// we now have fixed frequency bins, but we want the true frequencies (which in general lie between bins)
// to do this we start by getting the phase differences between frames
fn get_phase_difference(fft_spectrum: &Vec<Vec<Complex<f32>>>) -> Vec<Vec<f32>> {
    let mut phase_diffs: Vec<Vec<f32>> = Vec::with_capacity(fft_spectrum.len());

    // first frame needs to be done manually
    let first_frame_diffs: Vec<f32> = fft_spectrum[0].iter().map(|c| c.arg()).collect();
    phase_diffs.push(first_frame_diffs);

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

        phase_diffs.push(frame_diffs);
    }

    return phase_diffs;
}

fn correct_phase_diffs(phase_diffs: Vec<Vec<f32>>, hop_size: usize) -> Vec<Vec<f32>> {
    // We remove the expected phase difference due to natural accumulation
    // We also wrap back to [-π, π]
    let mut corrected_phase_diffs: Vec<Vec<f32>> = Vec::with_capacity(phase_diffs.len());

    for diff in phase_diffs.iter() {
        let corrected_diff: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, &observed)| {
                let expected = (hop_size as f32) * 2.0 * PI * (i as f32) / (diff.len() as f32);
                phase_vocoder_helpers::modulo(observed - expected + PI, 2.0 * PI) - PI
            })
            .collect();
        corrected_phase_diffs.push(corrected_diff);
    }

    return corrected_phase_diffs;
}

// gets the true frequency bins from the given bins and the phase differences
fn get_true_frequency(phase_differences: Vec<Vec<f32>>, hop_size: usize) -> Vec<Vec<f32>> {
    let mut true_frequencies: Vec<Vec<f32>> = Vec::with_capacity(phase_differences.len());
    for diff in phase_differences.iter() {
        let true_freq_frame: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, pd)| {
                let bin_freq = 2.0 * PI * (i as f32) / (diff.len() as f32);
                bin_freq + pd / (hop_size as f32)
            })
            .collect();
        true_frequencies.push(true_freq_frame);
    }

    return true_frequencies;
}

fn get_cumulative_phases(true_frequencies: Vec<Vec<f32>>, hop_out: usize) -> Vec<Vec<f32>> {
    let mut final_phases: Vec<Vec<f32>> = Vec::with_capacity(true_frequencies.len());

    // first frame needs to be done manually
    let first_phases: Vec<f32> = true_frequencies[0]
        .iter()
        .map(|f| f * (hop_out as f32))
        .collect();
    final_phases.push(first_phases);

    for (i, freq_frame) in true_frequencies.iter().skip(1).enumerate() {
        let phase_diff_frame: Vec<f32> = freq_frame.iter().map(|f| f * (hop_out as f32)).collect();
        let cumulative_phase_frame =
            phase_vocoder_helpers::elementwise_add(&final_phases[i], &phase_diff_frame);
        final_phases.push(cumulative_phase_frame);
    }
    return final_phases;
}

// End section 2

// Section 3: Synthesis

fn inverse_fft(fft_spectrum: &Vec<Vec<Complex<f32>>>, phases: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut output_frames: Vec<Vec<f32>> = Vec::with_capacity(fft_spectrum.len());
    for (fft_frame, phase_frame) in fft_spectrum.iter().zip(phases.iter()) {
        let rotated_frame = phase_vocoder_helpers::produce_output_frame(fft_frame, phase_frame);
        let output_frame = apply_fft_to_frame(&rotated_frame, FftDirection::Inverse)
            // take the real part
            .iter()
            .map(|c| c.re / (rotated_frame.len() as f32))
            .collect();
        output_frames.push(output_frame);
    }
    return output_frames;
}

fn overlap_add_frames(
    output_frames: Vec<Vec<f32>>,
    hop_out: usize,
    apply_window: bool,
) -> Vec<f32> {
    let mut raw_frames: Vec<Vec<f32>> = Vec::with_capacity(output_frames.len());
    for frame in output_frames {
        let output_frame = if apply_window {
            phase_vocoder_helpers::apply_hanning_window(&frame, hop_out)
        } else {
            frame
        };
        raw_frames.push(output_frame);
    }
    // set up the vectors
    let frame_length = raw_frames[0].len();
    let waveform_size = (raw_frames.len() - 1) * hop_out + frame_length;
    let mut overlapped_waveform: Vec<f32> = vec![0.0; waveform_size];

    // first iteration manually
    let range = 0..frame_length;
    let overlap_slice = phase_vocoder_helpers::elementwise_add(
        &overlapped_waveform[range.clone()],
        raw_frames[0].as_slice(),
    );
    let _ = overlapped_waveform.splice(range, overlap_slice);

    // overlap add by hop_out hop size
    for (i, frame) in raw_frames.iter().enumerate().skip(1) {
        let time_index = i * hop_out;
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
    println!("number of output samples is {}", resampled_waveform.len());
    return resampled_waveform;
}

// End section 4

mod phase_vocoder_helpers {
    use crate::wave::{StereoWave, WaveForm};
    use apodize::hanning_iter;
    use rustfft::num_complex::Complex;

    pub fn elementwise_add(v1: &[f32], v2: &[f32]) -> Vec<f32> {
        v1.into_iter().zip(v2).map(|(a, b)| a + b).collect()
    }

    pub fn modulo(a: f32, b: f32) -> f32 {
        ((a % b) + b) % b
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
        let example_fft: Vec<Vec<Complex<f32>>> = vec![
            vec![Complex { re: 1.0, im: 0.0 }, Complex { re: 1.0, im: 1.0 }],
            vec![Complex { re: 0.0, im: 1.0 }, Complex { re: 0.0, im: 1.0 }],
        ];
        let output_vec = vec![vec![0.0, PI / 4.0], vec![PI / 2.0, PI / 4.0]];
        let phase_diffs = get_phase_difference(&example_fft);
        assert_eq!(output_vec, phase_diffs);
    }

    #[test]
    fn modulo_test() {
        let x: f32 = phase_vocoder_helpers::modulo(3.0 * PI, 2.0 * PI);
        assert!((x - PI).abs() < 0.0001);
    }

    // End Test Section 2

    // Test Section 3: Synthesis

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
        let input: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let output: Vec<f32> = vec![1.0, 6.0, 8.0, 6.0];
        assert_eq!(output, overlap_add_frames(input, hop, false));
    }

    // End Test Section 3

    // Test Section 4 Resampling

    // End Test Section 4

    // Overall Tests
}
