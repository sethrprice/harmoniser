# Introduction
An in-progress project building a low-latency harmoniser in Rust.

# Progress 
Currently we have a phase vocoder algorithm that can shift pitch, and it does it pretty slowly. Next steps are:

* improve the performance of the pitch shifting. 
* create a simple user interface with intention to use as plugin
* turn phase vocoder algorithm into harmoniser algorithm
  * formant shifting, source and filter models for voice (pitch vs timbre), cepstral windowing
  * phase consistency
  * handle transients
  * naturalness: vibrato, tone
  * dynamic hop/frame variation for different pitches, maybe even ML