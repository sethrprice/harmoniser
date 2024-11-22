use rustfft::num_traits::Num;

// TODO use Option instead of an empty vec
#[derive(PartialEq, Debug)]
pub struct FramedVec<T> {
    vec: Vec<T>,
    hop_size: usize,
    frame_size: usize,
    n_frames: Option<usize>,
}

impl<T: Clone + Num> FramedVec<T> {
    pub fn new(samples: Vec<T>, frame_size: usize, hop_size: usize) -> Self {
        let n_frames = (samples.len() + hop_size - 1) / hop_size;
        let mut framed_vec: Vec<T> = Vec::with_capacity(n_frames * frame_size);

        for i in 0..n_frames {
            let start = i * hop_size;
            let end = start + frame_size;

            if end <= samples.len() {
                let frame = &samples[start..end];
                framed_vec.extend_from_slice(frame);
            } else {
                // handle padding
                let frame = &samples[start..];
                framed_vec.extend_from_slice(frame);
                framed_vec.extend(std::iter::repeat(T::zero()).take(frame_size - frame.len()));
            }
        }

        Self {
            vec: framed_vec,
            hop_size,
            frame_size,
            n_frames: Some(n_frames),
        }
    }

    pub fn with_capacity(capacity: usize, frame_size: usize, hop_size: usize) -> Self {
        Self {
            vec: Vec::with_capacity(capacity),
            frame_size,
            hop_size,
            n_frames: None,
        }
    }

    pub fn with_capacity_like(other: &Self) -> Self {
        Self::with_capacity(other.len(), other.frame_size, other.hop_size)
    }

    pub fn frame(&self, frame_idx: usize) -> Option<&[T]> {
        let start = frame_idx * self.frame_size;
        let end = start + self.frame_size;
        self.vec.get(start..end)
    }

    pub fn extend_from_slice(&mut self, slice: &[T]) {
        self.vec.extend_from_slice(slice);
        self.n_frames = Some((self.vec.len() + self.hop_size - 1) / self.hop_size);
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn number_of_frames(&self) -> usize {
        self.n_frames.unwrap_or(0)
    }

    #[cfg(test)]
    pub fn clone_vec(&self) -> Vec<T> {
        self.vec.clone()
    }
}
