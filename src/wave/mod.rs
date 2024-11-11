use hound::{WavReader, WavSpec, WavWriter};
use std::error::Error;

mod wavemetadata;
pub use wavemetadata::WaveMetaData;

mod stereowave;
pub use stereowave::StereoWave;

// i24::MAX needs to be defined manually because Rust has no native i24 type
const I24_MAX: f32 = ((1 << 23) - 1) as f32;

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub enum ChannelType {
    #[default]
    Mono,
    Stereo,
}

#[derive(Default, Debug, Copy, Clone, PartialEq)]
pub enum AudioFormat {
    #[default]
    PCM,
    IEEEFloat,
}

pub type WaveForm = Vec<f32>;

// we always write the waveform as a linear Vec, whether it's mono or stereo. We need to process a stereo signal as such.
#[derive(Default)]
pub struct Wave {
    waveform: WaveForm,
    metadata: WaveMetaData,
}

impl Wave {
    pub fn new(
        waveform: WaveForm,
        sample_rate: u32,
        channels: ChannelType,
        bit_depth: u16,
        data_length: usize,
        audio_format: AudioFormat,
    ) -> Self {
        let metadata =
            WaveMetaData::new(sample_rate, channels, bit_depth, data_length, audio_format);

        Self { waveform, metadata }
    }

    pub fn write_to_wav_file(&self, file_path: &str) -> Result<(), Box<dyn Error>> {
        println!("Writing to file...");

        // build a spec object
        let spec = WavSpec {
            channels: {
                match self.metadata.get_channels() {
                    ChannelType::Mono => 1,
                    ChannelType::Stereo => 2,
                }
            },
            sample_rate: self.metadata.get_sample_rate(),
            bits_per_sample: self.metadata.get_bit_depth(),
            sample_format: {
                match self.metadata.get_audio_format() {
                    AudioFormat::PCM => hound::SampleFormat::Int,
                    AudioFormat::IEEEFloat => hound::SampleFormat::Float,
                }
            },
        };

        //create a new writer object
        let mut writer = WavWriter::create(file_path, spec)?;

        // iterate over samples, change them to i32, and write them to file
        match self.metadata.get_bit_depth() {
            16 => {
                for &sample in &self.waveform {
                    let int_sample = (sample * i16::MAX as f32) as i16;
                    writer.write_sample(int_sample)?;
                }
            }
            24 => {
                for &sample in &self.waveform {
                    let int_sample = (sample * I24_MAX as f32) as i32;
                    writer.write_sample(int_sample)?;
                }
            }
            32 => {
                for &sample in &self.waveform {
                    let int_sample = (sample * i32::MAX as f32) as i32;
                    writer.write_sample(int_sample)?;
                }
            }

            _ => return Err("Error: unsupported bit depth for writing.".into()),
        }

        // finalise writing and close the file
        writer.finalize()?;

        Ok(())
    }

    pub fn get_waveform(&self) -> &WaveForm {
        &self.waveform
    }

    pub fn get_metadata(&self) -> WaveMetaData {
        self.metadata
    }
}

impl TryFrom<&str> for Wave {
    type Error = Box<dyn Error>;

    fn try_from(file_path: &str) -> Result<Self, Self::Error> {
        println!("Loading from file...");
        // open the file with hound
        let mut reader = WavReader::open(file_path)?;

        // read the metadata
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = match spec.channels {
            1 => ChannelType::Mono,
            2 => ChannelType::Stereo,
            0 => return Err("Invalid WAV file: 0 channels specified".into()),
            c => return Err(format!("Unsupported channel configuration: {} channels", c).into()),
        };
        let bit_depth = spec.bits_per_sample;
        let audio_format = match spec.sample_format {
            hound::SampleFormat::Int => AudioFormat::PCM,
            hound::SampleFormat::Float => AudioFormat::IEEEFloat,
        };

        // read the sample data, normalise, and convert to f32
        let samples: Vec<f32> = reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / I24_MAX))
            .collect::<Result<Vec<f32>, _>>()?;

        let data_length = samples.len();

        // create Wave instance

        let wave_metadata =
            WaveMetaData::new(sample_rate, channels, bit_depth, data_length, audio_format);

        let wave = Wave {
            waveform: samples,
            metadata: wave_metadata,
        };

        Ok(wave)
    }
}

impl core::fmt::Debug for Wave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "wave format: {}, bit depth: {}, channels: {:?}, sample rate: {}",
            self.waveform[self.metadata.get_data_length() / 2],
            self.metadata.get_bit_depth(),
            self.metadata.get_channels(),
            self.metadata.get_sample_rate()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_wav_metadata() {
        let test_wave = Wave::try_from("minimal.wav").unwrap();
        assert_eq!(test_wave.metadata.get_audio_format(), AudioFormat::PCM);
        assert_eq!(test_wave.metadata.get_bit_depth(), 24 as u16);
        assert_eq!(test_wave.metadata.get_channels(), ChannelType::Mono);
        assert_eq!(test_wave.metadata.get_sample_rate(), 44100 as u32);
    }

    #[test]
    fn load_wav_waveform() {
        let test_wave = Wave::try_from("minimal.wav").unwrap();
        let test_vec: Vec<f32> = vec![-0.3, 0.4, 0.5];
        for (v1, v2) in test_vec.iter().zip(test_wave.waveform.iter()) {
            assert!((v1 - v2).abs() < 0.001);
        }
    }

    #[test]
    fn load_save_reload_wave() {
        let test_wave = Wave::try_from("minimal.wav").unwrap();
        let filepath = "minimal_2.wav";
        test_wave.write_to_wav_file(filepath).unwrap();
        let test_wave_2 = Wave::try_from(filepath).unwrap();
        let test_vec: Vec<f32> = vec![-0.3, 0.4, 0.5];
        for (v1, v2) in test_vec.iter().zip(test_wave_2.waveform.iter()) {
            assert!((v1 - v2).abs() < 0.001);
        }
        std::fs::remove_file(filepath).unwrap();
    }
}
