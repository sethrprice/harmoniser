mod wave;
use wave::Wave;

mod phasevocoder;
use phasevocoder::PhaseVocoder;

use std::error::Error;
use std::time::Instant;

const FRAME_SIZE: usize = 1024;

fn main() {
    if let Err(e) = run() {
        eprintln!("Application error: {e}")
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let wave = Wave::try_from("test_wavs/test.wav")?;
    println!("{wave:?}");
    let phase_vocoder = PhaseVocoder::new(&wave, FRAME_SIZE);
    let now = Instant::now();
    let new_wave = phase_vocoder.shift_signal(3);
    let elapsed = now.elapsed();
    println!("Elapsed: {:.2?}", elapsed);
    println!("{new_wave:?}");
    new_wave.write_to_wav_file("test_wavs/test_out.wav")?;
    Ok(())
}
