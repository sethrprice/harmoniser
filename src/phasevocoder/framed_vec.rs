use rustfft::num_traits::Num;

// TODO use Option instead of an empty vec
#[derive(PartialEq, Debug)]
pub struct FramedVec<T> {
    vec: Vec<T>,
    hop_size: usize,
    frame_size: usize,
    indices: Vec<(usize, usize)>,
}

pub struct FrameIter<'a, T> {
    vec: &'a [T],
    indices: std::slice::Iter<'a, (usize, usize)>,
}

impl<T: Clone + Num> FramedVec<T> {
    pub fn new(vec: Vec<T>, frame_size: usize, hop_size: usize) -> Self {
        // calculate the number of frames and the length of the vector + the padding
        let num_frames = (vec.len() + hop_size - 1) / hop_size;
        let padded_length = (num_frames.saturating_sub(1)) * hop_size + frame_size;

        // pad the vector so we always frame all our samples
        let mut padded_vec = vec.clone();
        padded_vec.resize(padded_length, T::zero());

        let indices: Vec<(usize, usize)> = (0..num_frames)
            .map(|i| (i * hop_size, i * hop_size + frame_size))
            .collect();

        Self {
            vec: padded_vec,
            hop_size,
            frame_size,
            indices,
        }
    }

    // TODO use Option instead of an empty vec
    pub fn new_empty_non_overlapped(length: usize, frame_size: usize) -> Self {
        let vec = Vec::with_capacity(length);
        let indices = vec![];
        Self {
            vec,
            hop_size: frame_size,
            frame_size,
            indices,
        }
    }

    pub fn push(&mut self, frame: Vec<T>) {
        for element in frame {
            self.vec.push(element);
        }
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn iter(&self) -> FrameIter<T> {
        FrameIter {
            vec: &self.vec,
            indices: self.indices.iter(),
        }
    }

    pub fn hop(&self) -> usize {
        self.hop_size
    }

    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    pub fn update_indices(&mut self) {
        let num_frames = (self.vec.len() + self.hop_size - 1) / self.hop_size;
        let padded_length = (num_frames - 1) * self.hop_size + self.frame_size;

        self.vec.resize(padded_length, T::zero());

        let indices: Vec<(usize, usize)> = (0..num_frames)
            .map(|i| (i * self.hop_size, i * self.hop_size + self.frame_size))
            .collect();

        self.indices = indices;
    }

    pub fn number_of_frames(&self) -> usize {
        self.indices.len()
    }

    #[cfg(test)]
    pub fn clone_vec(&self) -> Vec<T> {
        self.vec.clone()
    }
}

impl<'a, T> Iterator for FrameIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|&(start, end)| &self.vec[start..end])
    }
}
