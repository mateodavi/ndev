mod network;

#[cfg(test)]
mod tests {
    use crate::network::{GeneticAlgorithm, Selection, SelectionType};

    #[test]
    fn xor() {
        let mut ga = GeneticAlgorithm::new(
            100,
            &[2, 4, 1],
            Selection::new(0.25f32, 0.04f32, SelectionType::Truncation),
            0.2f32,
        );
        ga.run(&[
            &[0f32, 0f32],
            &[0f32, 1f32],
            &[1f32, 0f32],
            &[1f32, 1f32],
        ], &[
            &[0f32],
            &[1f32],
            &[1f32],
            &[0f32],
        ]);

        assert_eq!(2 + 2, 4);
    }
}
