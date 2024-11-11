use super::WaveForm;
use std::error::Error;

pub struct StereoWave {
    left: WaveForm,
    right: WaveForm,
}

impl StereoWave {
    pub fn new(left: Vec<f32>, right: Vec<f32>) -> Result<Self, Box<dyn Error>> {
        if left.len() != right.len() {
            Err("Both channels in a stereo wave must be the same length".into())
        } else {
            Ok(Self { left, right })
        }
    }

    pub fn get_channel_length(&self) -> usize {
        self.left.len()
    }

    pub fn get_left_channel(&self) -> &WaveForm {
        &self.left
    }

    pub fn get_right_channel(&self) -> &WaveForm {
        &self.right
    }
}

impl From<StereoWave> for WaveForm {
    fn from(stereo_wave: StereoWave) -> Self {
        let StereoWave { left, right } = stereo_wave;
        let output_wave = stereowave_helpers::interleave(left, right);
        return output_wave;
    }
}

mod stereowave_helpers {
    pub fn interleave<T: Clone>(vec1: Vec<T>, vec2: Vec<T>) -> Vec<T> {
        vec1.into_iter()
            .zip(vec2)
            .flat_map(|(a, b)| vec![a, b])
            .collect()
    }
}
