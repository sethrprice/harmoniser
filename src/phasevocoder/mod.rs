use crate::wave::{ChannelType, StereoWave, Wave, WaveForm, WaveMetaData};
use apodize::hanning_iter;
use phase_vocoder_helpers::split_stereo_wave;
use rust_interpolation::monointerp;
use rustfft::{algorithm::Radix4, num_complex::Complex, Fft, FftDirection};
use std::f32::consts::PI;
mod framed_vec;
use framed_vec::FramedVec;
use rayon::prelude::*;
use std::time::Instant;

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
        let now_a = Instant::now();
        let frames = generate_frames(waveform.to_vec(), self.frame_size, self.hop_size);
        println!("number of samples = {}", frames.len());
        println!("number of frames = {}", frames.number_of_frames());
        let now_fft = Instant::now();
        let fft_frames = fft(frames, self.frame_size, self.hop_size);
        let elapsed_fft = now_fft.elapsed();
        println!("FFT took {elapsed_fft:.2?}");
        let elapsed_a = now_a.elapsed();
        println!("analysis took {elapsed_a:.2?}");
        // End analysis

        // Processing
        let now_p = Instant::now();
        let phase_differences = get_phase_difference(&fft_frames);
        let corrected_phase_differences = correct_phase_diffs(&phase_differences, self.hop_size);
        let true_frequencies = get_true_frequency(&corrected_phase_differences, self.hop_size);
        let final_phases = get_cumulative_phases(&true_frequencies, hop_out);
        let elapsed_p = now_p.elapsed();
        println!("processing took {elapsed_p:.2?}");
        // End processing

        // Synthesis
        let now_s = Instant::now();
        let output_frames = inverse_fft(&fft_frames, &final_phases);
        let elapsed_ifft = now_s.elapsed();
        println!("IFFT took {elapsed_ifft:.2?}");
        let overlapped_waveform = overlap_add_frames(&output_frames, hop_out, true);
        let elapsed_s = now_s.elapsed();
        println!("synthesis took {elapsed_s:.2?}");
        // End synthesis

        // Resampling
        let now_r = Instant::now();
        let output_waveform = resample(overlapped_waveform, alpha);
        let elapsed_r = now_r.elapsed();
        println!("resampling took {elapsed_r:.2?}");
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

pub fn generate_hanning_window(length: usize, hop: usize, normalise: bool) -> Vec<f32> {
    // normalisation of window compensates for overlap adding
    let normalisation = if normalise {
        (((length / hop) / 2) as f32).sqrt()
    } else {
        1.
    };

    // generate a symmetrical Hanning window
    hanning_iter(length * 2 + 1)
        .skip(1)
        .step_by(2)
        .map(|v| (v as f32) / normalisation)
        .collect()
}

pub fn apply_hanning_window(frame: &[f32], window: &[f32]) -> Vec<f32> {
    assert_eq!(frame.len(), window.len());

    frame
        .iter()
        .zip(window.iter())
        .map(|(x, w)| x * w)
        .collect()
}

fn apply_fft_to_frame<T>(frame: &[T], fft_engine: &dyn Fft<f32>) -> Vec<Complex<f32>>
where
    T: Into<Complex<f32>> + Copy,
{
    let mut buffer: Vec<Complex<f32>> = frame.iter().map(|&x| x.into()).collect();
    fft_engine.process(&mut buffer);
    return buffer;
}

fn fft(frames: FramedVec<f32>, frame_size: usize, hop: usize) -> FramedVec<Complex<f32>> {
    // precompute window
    let window = generate_hanning_window(frame_size, hop, true);

    // precompute the default value for out-of-bounds indices
    let default: Vec<f32> = vec![0.; frame_size];

    // precompute frame info
    let number_of_frames = frames.number_of_frames();
    let total_fft_size = number_of_frames * frame_size;

    // Preallocate contiguous storage for FFT results
    let mut fft_frames: Vec<Complex<f32>> = vec![Complex { re: 0.0, im: 0.0 }; total_fft_size];

    // precompute fft engine
    let fft_engine = Radix4::new(frame_size, FftDirection::Forward);

    fft_frames
        .par_chunks_mut(frame_size) // Split the contiguous buffer into chunks
        .enumerate()
        .for_each(|(i, fft_frame)| {
            let frame = frames.frame(i).unwrap_or(&default);
            let windowed_frame = apply_hanning_window(frame, &window);

            // Convert windowed frame to complex and copy into fft_frame
            fft_frame.copy_from_slice(&apply_fft_to_frame(&windowed_frame, &fft_engine));
        });

    return FramedVec::new(fft_frames, frame_size, frame_size);
}

// End section 1

// Section 2: Processing

// we now have fixed frequency bins, but we want the true frequencies (which in general lie between bins)
// to do this we start by getting the phase differences between frames
fn get_phase_difference(fft_spectrum: &FramedVec<Complex<f32>>) -> FramedVec<f32> {
    let mut phase_diffs: FramedVec<f32> = FramedVec::with_capacity(
        fft_spectrum.len(),
        fft_spectrum.frame_size(),
        fft_spectrum.frame_size(),
    );

    let number_of_frames = fft_spectrum.number_of_frames();

    // first frame needs to be done manually
    let first_frame_diffs: Vec<f32> = fft_spectrum
        .frame(0)
        .expect("frame index out of bounds")
        .iter()
        .map(|c| c.arg())
        .collect();
    phase_diffs.extend_from_slice(&first_frame_diffs);

    // iterate over all consecutive frames and take the phase difference between them
    for i in 0..(number_of_frames - 1) {
        let prev_frame = fft_spectrum.frame(i).expect("frame index out of bounds");
        let curr_frame = fft_spectrum
            .frame(i + 1)
            .expect("frame index out of bounds");
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
        phase_diffs.extend_from_slice(&frame_diffs);
    }

    return phase_diffs;
}

fn correct_phase_diffs(phase_diffs: &FramedVec<f32>, hop_size: usize) -> FramedVec<f32> {
    // We remove the expected phase difference due to natural accumulation
    // We also wrap back to [-π, π]
    let mut corrected_phase_diffs: FramedVec<f32> = FramedVec::with_capacity_like(phase_diffs);

    let number_of_frames = phase_diffs.number_of_frames();

    for i in 0..number_of_frames {
        let diff = phase_diffs.frame(i).expect("frame index out of bounds");
        let corrected_diff: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, &observed)| {
                let expected = (hop_size as f32) * 2.0 * PI * (i as f32) / (diff.len() as f32);
                phase_vocoder_helpers::modulo(observed - expected + PI, 2.0 * PI) - PI
            })
            .collect();
        corrected_phase_diffs.extend_from_slice(&corrected_diff);
    }

    return corrected_phase_diffs;
}

// gets the true frequency bins from the given bins and the phase differences
fn get_true_frequency(phase_differences: &FramedVec<f32>, hop_size: usize) -> FramedVec<f32> {
    let mut true_frequencies: FramedVec<f32> = FramedVec::with_capacity_like(phase_differences);

    let number_of_frames = phase_differences.number_of_frames();

    for i in 0..number_of_frames {
        let diff = phase_differences
            .frame(i)
            .expect("frame index out of bounds");
        let true_freq_frame: Vec<f32> = diff
            .iter()
            .enumerate()
            .map(|(i, pd)| {
                let bin_freq = 2.0 * PI * (i as f32) / (diff.len() as f32);
                bin_freq + pd / (hop_size as f32)
            })
            .collect();
        true_frequencies.extend_from_slice(&true_freq_frame);
    }

    return true_frequencies;
}

fn get_cumulative_phases(true_frequencies: &FramedVec<f32>, hop_out: usize) -> FramedVec<f32> {
    let mut final_phases: FramedVec<f32> = FramedVec::with_capacity_like(true_frequencies);

    let number_of_frames = true_frequencies.number_of_frames();

    for i in 0..number_of_frames {
        let freq_frame: Vec<f32> = true_frequencies
            .frame(i)
            .expect("frame index out of bounds")
            .iter()
            .map(|f| f * (hop_out as f32))
            .collect();
        let default: Vec<f32> = vec![0.; freq_frame.len()];
        let previous_frame = final_phases.frame(i.saturating_sub(1)).unwrap_or(&default);
        let cumulative_phase_frame =
            phase_vocoder_helpers::elementwise_add(previous_frame, &freq_frame);
        final_phases.extend_from_slice(&cumulative_phase_frame);
    }

    return final_phases;
}

// End section 2

// Section 3: Synthesis

fn inverse_fft(fft_spectrum: &FramedVec<Complex<f32>>, phases: &FramedVec<f32>) -> FramedVec<f32> {
    let number_of_frames = fft_spectrum.number_of_frames();
    let frame_size = fft_spectrum.frame_size();
    let total_ifft_size = number_of_frames * frame_size;

    let mut ifft_frames: Vec<f32> = vec![0.; total_ifft_size];

    let ifft_engine = Radix4::new(fft_spectrum.frame_size(), FftDirection::Inverse);

    ifft_frames
        .par_chunks_mut(frame_size) // Split the contiguous buffer into chunks
        .enumerate()
        .for_each(|(i, ifft_frame)| {
            let frame = fft_spectrum.frame(i).expect("frame index out of bounds");
            let phase_frame = phases.frame(i).expect("frame index out of bounds");
            let rotated_frame = phase_vocoder_helpers::produce_output_frame(frame, phase_frame);
            let output_frame: Vec<f32> = apply_fft_to_frame(&rotated_frame, &ifft_engine)
                // take the real part and normalise
                .iter()
                .map(|c| c.re / (rotated_frame.len() as f32))
                .collect();

            // Convert windowed frame to complex and copy into fft_frame
            ifft_frame.copy_from_slice(&output_frame);
        });
    return FramedVec::new(ifft_frames, frame_size, frame_size);
}

// TODO I expect this function will need further optimising
fn overlap_add_frames(
    output_frames: &FramedVec<f32>,
    hop_out: usize,
    apply_window: bool,
) -> Vec<f32> {
    let mut raw_frames: FramedVec<f32> = FramedVec::with_capacity_like(output_frames);
    println!("frame size is {}", output_frames.frame_size());

    let window = generate_hanning_window(output_frames.frame_size(), hop_out, true);

    let number_of_frames = output_frames.number_of_frames();

    for i in 0..number_of_frames {
        let frame = output_frames.frame(i).expect("frame index out of bounds");
        let output_frame = if apply_window {
            apply_hanning_window(&frame, &window)
        } else {
            frame.to_vec()
        };
        raw_frames.extend_from_slice(&output_frame);
    }

    // set up the vectors
    let frame_length = raw_frames.frame_size();
    let new_waveform_size = (raw_frames.number_of_frames() - 1) * hop_out + frame_length;
    let mut overlapped_waveform: Vec<f32> = vec![0.; new_waveform_size];

    for i in 0..number_of_frames {
        let start = i * hop_out;
        let end = start + frame_length;
        let frame = raw_frames.frame(i).expect("frame index out of bounds");
        let overlap_slice =
            phase_vocoder_helpers::elementwise_add(&overlapped_waveform[start..end], &frame);
        let _ = overlapped_waveform.splice(start..end, overlap_slice);
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
        for i in 0..frames.number_of_frames() {
            let frame = frames.frame(i).expect("frame index out of bounds");
            assert_eq!(frame, output[i]);
        }
    }

    #[test]
    fn hanning_window_generation() {
        let hanning_test: Vec<f32> = vec![
            0.0, 0.11697778, 0.41317591, 0.75, 0.96984631, 0.96984631, 0.75, 0.41317591,
            0.11697778, 0.0,
        ];
        let hanning_generated = generate_hanning_window(10, 1, false);
        for (v1, v2) in hanning_test.iter().zip(hanning_generated.iter()) {
            assert!(v1 - v2 < 0.0001);
        }
    }

    #[test]
    fn fft_test() {
        let v: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0];
        let fft_engine = Radix4::new(8, FftDirection::Forward);
        let fft_vec: Vec<f32> = apply_fft_to_frame(&v, &fft_engine)
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
        let corrected_pds = correct_phase_diffs(&input, 1).clone_vec();
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
        let true_freq = get_true_frequency(&phase_differences, 2).clone_vec();
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
        let cumulative_phases = get_cumulative_phases(&true_freqs, 1).clone_vec();
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
        assert_eq!(output, overlap_add_frames(&input, hop, false));
    }

    // End Test Section 3

    // Test Section 4 Resampling

    // End Test Section 4

    // Overall Tests
}
