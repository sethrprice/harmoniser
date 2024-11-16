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

impl<T> FramedVec<T> {
    pub fn new(vec: Vec<T>, frame_size: usize, hop_size: usize) -> Self {
        let indices: Vec<(usize, usize)> = (0..vec.len())
            .map(|i| (i * hop_size, i * hop_size + frame_size))
            // TODO think about what happens to any leftover tail -- do I need to account for this?
            .take_while(|&(i, _)| i * hop_size + frame_size < vec.len())
            .collect();

        Self {
            vec,
            hop_size,
            frame_size,
            indices,
        }
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn non_overlapping_len(&self) -> usize {
        let n_frames = (self.len() - self.frame_size) / self.hop_size;
        let n_samples = n_frames * self.frame_size;
        return n_samples;
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
}

impl<'a, T> Iterator for FrameIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|&(start, end)| &self.vec[start..end])
    }
}
