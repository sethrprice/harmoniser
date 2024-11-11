use super::{AudioFormat, ChannelType};

#[derive(Default, Debug, Copy, Clone)]
pub struct WaveMetaData {
    sample_rate: u32,
    channels: ChannelType,
    bit_depth: u16,
    data_length: usize,
    audio_format: AudioFormat,
}

impl WaveMetaData {
    pub fn new(
        sample_rate: u32,
        channels: ChannelType,
        bit_depth: u16,
        data_length: usize,
        audio_format: AudioFormat,
    ) -> Self {
        Self {
            sample_rate,
            channels,
            bit_depth,
            data_length,
            audio_format,
        }
    }

    pub fn get_sample_rate(&self) -> u32 {
        self.sample_rate
    }

    pub fn get_channels(&self) -> ChannelType {
        self.channels
    }

    pub fn get_bit_depth(&self) -> u16 {
        self.bit_depth
    }

    pub fn get_data_length(&self) -> usize {
        self.data_length
    }

    pub fn get_audio_format(&self) -> AudioFormat {
        self.audio_format
    }
}
