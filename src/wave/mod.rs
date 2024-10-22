use hound::{WavReader, WavSpec, WavWriter};
use std::error::Error;

#[derive(Default, Debug)]
pub enum ChannelType{
    #[default]
    Mono, 
    Stereo
}

#[derive(Default, Debug)]
pub enum AudioFormat{
    #[default]
    PCM,
    IEEEFloat,
}

#[derive(Default, Debug)]
pub struct Wave{
    waveform: Vec<f32>, 
    sample_rate: u32, 
    channels: ChannelType,
    bit_depth: u16, 
    data_length: usize, 
    audio_format: AudioFormat
}


impl Wave{

    pub fn write_to_wav_file(&self, file_path: &str)->Result<(), Box<dyn Error>>{
        
        // build a spec object
        let spec = WavSpec{
            channels: {
                match self.channels{
                    ChannelType::Mono => 1, 
                    ChannelType::Stereo => 2,
                }
            }, 
            sample_rate: self.sample_rate, 
            bits_per_sample: self.bit_depth, 
            sample_format: {
                match self.audio_format{
                    AudioFormat::PCM => hound::SampleFormat::Int, 
                    AudioFormat::IEEEFloat => hound::SampleFormat::Float,
                }
            }
        };

        //create a new writer object
        let mut writer = WavWriter::create(file_path, spec)?;

        // iterate over samples, change them to i32, and write them to file
        match self.bit_depth{
            16 => {
                for &sample in &self.waveform{
                    let int_sample = (sample * i16::MAX as f32) as i16;
                    writer.write_sample(int_sample)?;
                }
            }, 
            24 => { 
                // this is complicated, but essentially truncates a 32-bit number to 24 to save space.
                // delete this and use 32 if it doesn't work. Figure out later.
                for &sample in &self.waveform{
                    let int_sample = (sample * (i32::MAX >> 8)as f32) as i32;
                    let bytes = int_sample.to_be_bytes();  // Get the bytes of the i32
                    writer.write_sample(((bytes[1] as i32) << 16) | ((bytes[2] as i32) << 8) | (bytes[3] as i32))?; 
                }
            }
            32 => {
                for &sample in &self.waveform{
                    let int_sample = (sample * i32::MAX as f32) as i32;
                    writer.write_sample(int_sample)?;
                }
            },

            _ => return Err("Error: unsupported bit depth for writing.".into())
        }
        

        // finalise writing and close the file
        writer.finalize()?;


        Ok(())
    }

}


impl TryFrom<&str> for Wave{
    type Error = Box<dyn Error>;

    fn try_from(file_path: &str) -> Result<Self, Self::Error> {
         // open the file with hound
        let mut reader = WavReader::open(file_path)?;

        // read the metadata
        let spec = reader.spec();
        let sample_rate = spec.sample_rate;
        let channels = match spec.channels{
            1 => ChannelType::Mono, 
            2 => ChannelType::Stereo, 
            0 => return Err("Invalid WAV file: 0 channels specified".into()),
            c => return Err(format!("Unsupported channel configuration: {} channels", c).into())
        };
        let bit_depth = spec.bits_per_sample;
        let audio_format = match spec.sample_format{
            hound::SampleFormat::Int => AudioFormat::PCM,
            hound::SampleFormat::Float => AudioFormat::IEEEFloat
        };

        // read the sample data, normalise, and convert to f32
        let samples: Vec<f32> = match spec.bits_per_sample{
            16 => {
                reader.samples::<i32>()
                    .map(|sample| {
                        sample.map(|v| v as f32 / (i16::MAX) as f32)
                    })
                    .collect::<Result<Vec<f32>, _>>()?
            },
            24 => {
                reader.samples::<i32>()
                    .map(|sample| {
                        sample.map(|v| (v >> 8) as f32 / (i32::MAX >> 8 )as f32)
                    })
                    .collect::<Result<Vec<f32>, _>>()?
            }, 
            _ => return Err("Unsupported bit depth: only 16-bit and 24-bit supported.".into())
        };
        
        let data_length = samples.len();

        // create Wave instance
        let wave = Wave{
            waveform: samples, 
            sample_rate, 
            channels, 
            bit_depth, 
            data_length, 
            audio_format
        };

        Ok(wave)
    }
}