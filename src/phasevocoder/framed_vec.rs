pub struct FramedVec<T> {
    vec: Vec<T>,
    hop_size: usize,
    frame_length: usize,
    indices: Vec<(usize, usize)>,
}

pub struct FrameIter<'a, T> {
    vec: &'a [T],
    indices: std::slice::Iter<'a, (usize, usize)>,
}

impl<T> FramedVec<T> {
    pub fn new(vec: Vec<T>, frame_length: usize, hop_size: usize) -> Self {
        let indices: Vec<(usize, usize)> = (0..vec.len())
            .map(|i| (i * hop_size, i * hop_size + frame_length))
            .take_while(|&(i, _)| i < vec.len())
            .collect();

        Self {
            vec,
            hop_size,
            frame_length,
            indices,
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
}

impl<'a, T> Iterator for FrameIter<'a, T> {
    type Item = &'a [T];

    fn next(&mut self) -> Option<Self::Item> {
        self.indices
            .next()
            .map(|&(start, end)| &self.vec[start..end])
    }
}
