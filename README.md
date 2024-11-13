# Introduction
An in-progress project building a low-latency harmoniser in Rust.

# Progress 
Currently we have a phase vocoder algorithm that can shift pitch, and it does it pretty slowly. Next steps are:

* improve the performance of the pitch shifting. `Vec<Vec<f32>>` -> `Vec<f32>`, this will be much more efficient
* create a simple user interface with intention to use as plugin
* turn phase vocoder algorithm into harmoniser algorithm