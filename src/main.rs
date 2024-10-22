
mod wave;
use wave::Wave;


fn main() {
    let wave = Wave::try_from("test.wav");
    println!("Wave = {wave:?}")
}
