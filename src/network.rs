use std::fmt;
use rand::{thread_rng, Rng};
use rand_distr::{Normal, Distribution};
use rayon::prelude::*;

#[allow(dead_code)]
pub struct Network {
    nodes: Vec<Node>,
    input_ids: Vec<usize>,
    output_ids: Vec<usize>,
    number_of_links: usize,
    fitness: f32,
}

#[allow(dead_code)]
struct Node {
    value: f32,
    links: Vec<(usize, f32)>,
    bias: f32,
    activation: Activation,
}

impl Node {
    fn new(value: f32, bias: f32, activation: Activation) -> Node {
        Node {
            value: value,
            links: Vec::new(),
            bias: bias,
            activation: activation,
        }
    }
}

#[derive(PartialEq)]
#[allow(dead_code)]
enum Activation {
    Relu,
    ReluLeaky,
    Tanh,
    Lstm,
    None,
}

#[allow(dead_code)]
impl Network {
    pub fn new(layers: &[usize]) -> Network {

        if layers.len() == 0 {
            panic!("Network must not be empty!");
        }
        if layers[layers.len() - 1] == 0 {
            panic!("Network must contain output nodes!")
        }

        let mut nodes: Vec<Node> = Vec::new();
        let mut this_layer_ids: Vec<usize> = Vec::new();
        let mut prev_layer_ids: Vec<usize>;
        let mut output_ids: Vec<usize> = Vec::new();
        let mut input_ids: Vec<usize> = Vec::new();
        let mut number_of_links = 0usize;
        let mut rng = thread_rng();

        // Input first
        for _ in 0..layers[0] {
            let node = Node::new(0f32, 0f32, Activation::None);
            let id = nodes.len();
            nodes.push(node);
            input_ids.push(id);
            this_layer_ids.push(id);
        }

        // Build network
        for layer in 1..layers.len(){
            prev_layer_ids = this_layer_ids;
            this_layer_ids = Vec::new();
            let normal = Normal::new(0f64, (2f64 / layers[layer] as f64).sqrt()).unwrap();
            for _ in 0..layers[layer] {
                let id = nodes.len();
                let activation;
                if layer == layers.len() - 1 {
                    activation = Activation::Tanh;
                } else {
                    activation = Activation::ReluLeaky;
                }
                let bias = 0f32; //normal.sample(&mut rng) as f32;
                let mut node = Node::new(0f32, bias, activation);
                for link_id in &prev_layer_ids {
                    node.links.push((*link_id, normal.sample(&mut rng) as f32));
                    number_of_links += 1;
                }
                nodes.push(node);
                if layer == layers.len() - 1 {
                    output_ids.push(id);
                }
                this_layer_ids.push(id);
            }
        }

        Network {
            nodes,
            input_ids,
            output_ids,
            number_of_links,
            fitness: 0f32,
        }
    }

    pub fn step(&mut self, inputs: &[f32]) -> Vec<f32> {

        if inputs.len() != self.input_ids.len() {
            panic!("Incorrect number of inputs!")
        }

        // Copy input
        for (i, input) in inputs.iter().enumerate() {
            self.nodes[i].value = *input;
        }

        // Step through network
        for i in inputs.len()..self.nodes.len() {
            let mut sum = 0f32;
            for link in &self.nodes[i].links {
                sum += self.nodes[link.0].value * link.1;
            }
            sum += self.nodes[i].bias;
            if self.nodes[i].activation == Activation::Relu {
                self.nodes[i].value = sum.max(0f32);
            } else if self.nodes[i].activation == Activation::ReluLeaky {
                self.nodes[i].value = if sum > 0f32 { sum } else { sum * 0.01f32 };
            } else if self.nodes[i].activation == Activation::Tanh {
                self.nodes[i].value = sum.tanh();
            } else {
                unimplemented!("To do, maybe");
            }
        }

        // Copy output
        let mut output: Vec<f32> = Vec::with_capacity(self.output_ids.len());
        for id in &self.output_ids {
            output.push(self.nodes[*id].value);
        }
        output
    }

    fn unwrap(&self) -> Vec<f32> {
        let mut unwrapped: Vec<f32> = Vec::with_capacity(self.number_of_links);
        for node in &self.nodes {
            for link in &node.links {
                unwrapped.push(link.1);
            }
            if node.activation != Activation::None {
                unwrapped.push(node.bias);
            }
        }
        unwrapped
    }

    fn copy(&mut self, links: &Vec<f32>) {
        let mut i = 0;

        // Copy link values into network
        for node in &mut self.nodes{
            for link in &mut node.links {
                if i >= links.len() {
                    panic!("Network contains too few links!");
                }
                link.1 = links[i];
                i += 1;
            }
            if node.activation != Activation::None {
                node.bias = links[i];
                i += 1;
            }
        }
    }
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for node in &self.nodes {
            write!(f, "{}, [", node.value).expect("Could not print!");
            for link in 0..node.links.len() {
                if link == node.links.len() - 1 {
                    write!(f, "{}", node.links[link].1).expect("Could not print!");
                } else {
                    write!(f, "{}, ", node.links[link].1).expect("Could not print!");
                }
            }
            writeln!(f, "]").expect("Could not print!");
        }
        Ok(())
    }
}

#[allow(dead_code)]
pub struct GeneticAlgorithm {
    networks: Vec<Network>,
    selection: Selection,
    mutation_probability: f32,
}

#[derive(PartialEq)]
#[allow(dead_code)]
pub struct Selection {
    selection_percentage: f32,
    elite_percentage: f32,
    selection_type: SelectionType,
}

#[allow(dead_code)]
impl Selection {
    pub fn new(selection_percentage: f32, elite_percentage: f32, selection_type: SelectionType) -> Selection {
        if selection_percentage < 0f32 || selection_percentage > 1f32 {
            panic!("Selection percentage must be between 0 and 1!");
        }
        if elite_percentage < 0f32 || elite_percentage > 1f32 || elite_percentage > selection_percentage {
            panic!("Elite percentage must be between 0 and 1, and less than selection percentage!");
        }

        Selection {
            selection_percentage,
            elite_percentage,
            selection_type,
        }
    }
}

#[derive(PartialEq)]
#[allow(dead_code)]
pub enum SelectionType {
    Truncation,
    Roulette,
}

#[allow(dead_code)]
impl GeneticAlgorithm {
    pub fn new(
        population_size: usize,
        network_topology: &[usize],
        selection: Selection,
        mutation_probability: f32
    ) -> GeneticAlgorithm {

        if mutation_probability < 0f32 || mutation_probability > 1f32 {
            panic!("Mutation probability must be between 0 and 1!");
        }

        let mut networks: Vec<Network> = Vec::with_capacity(population_size);

        for _ in 0..population_size {
            let network = Network::new(network_topology);
            networks.push(network);
        }

        GeneticAlgorithm {
            networks,
            selection,
            mutation_probability,
        }
    }

    pub fn run(&mut self, inputs: &'static [&[f32]], expecteds: &'static [&[f32]]) {
        if inputs.len() != expecteds.len() {
            panic!("Input and output data are not the same length.");
        }

        let mut generation = 0;
        for i in 0..10000 {
            let fitness = self.evaluate(inputs, expecteds);
            if fitness > 0.999f32 {
                break;
            }
            if i % 100 == 0 {
                println!("{}: {}", generation, self.networks[0].fitness);
            }

            self.optimize();
            generation += 1;
        }

        println!("{}: {}", generation, self.networks[0].fitness);
        for i in 0..inputs.len() {
            let output = self.networks[0].step(inputs[i]);
            println!("{:?} {:?}", inputs[i], output)
        }
        let unwrapped = self.networks[0].unwrap();
        println!("{:?}", unwrapped);
    }

    pub fn evaluate(&mut self, inputs: &'static [&[f32]], expecteds: &'static [&[f32]]) -> f32 {
        if inputs.len() != expecteds.len() {
            panic!("Input and output data are not the same length.");
        }

        self.networks.par_iter_mut().for_each(|network| {
            network.fitness = 0f32;
            for j in 0..inputs.len() {
                let output = network.step(inputs[j]);
                for (k, output_value) in output.iter().enumerate() {
                    network.fitness += (expecteds[j][k] - output_value).abs();
                }
            }
            network.fitness = 2f32 / (network.fitness.exp() + 1f32);
        });

        // Sort by fitness from highest to lowest
        self.networks.sort_unstable_by(
            |a, b| b.fitness.partial_cmp(&a.fitness).unwrap()
        );
        self.networks[0].fitness
    }

    fn optimize(&mut self) {
        let elite_amount = (self.networks.len() as f32 * self.selection.elite_percentage) as usize;
        let mut next_networks: Vec<Vec<f32>> = vec!(vec!(); self.networks.len() - elite_amount);
        let selected_ids = self.selection();
        self.crossover(&mut next_networks, selected_ids, elite_amount);
        self.mutation(&mut next_networks, elite_amount);
    }

    fn selection(&mut self) -> Vec<usize> {
        let selection_amount = (self.networks.len() as f32 * self.selection.selection_percentage) as usize;
        let mut selected_ids: Vec<usize> = Vec::with_capacity(selection_amount);

        // Perform selection
        if self.selection.selection_type == SelectionType::Truncation {
            for i in 0..selection_amount {
                selected_ids.push(i);
            }
        } else {
            unimplemented!("Unimplemented selection type!");
        }

        selected_ids
    }

    fn crossover(&mut self, next_networks: &mut Vec<Vec<f32>>, selected_ids: Vec<usize>, elite_amount: usize) {
        
        next_networks.par_iter_mut().for_each(|network| {
            for _ in 0..(self.networks.len() - elite_amount) {
                let mut rng = thread_rng();

                // Pick two parents from selection pool
                let id0 = rng.gen_range(0, selected_ids.len());
                let id1 = {
                    let id = rng.gen_range(0, selected_ids.len() - 1);
                    if id >= id0 { id + 1 } else { id }
                };

                // Unwrap
                let mut unwrapped0 = self.networks[id0].unwrap();
                let unwrapped1 = self.networks[id1].unwrap();
                
                // Pick crossover points
                let p0 = rng.gen_range(0, (unwrapped0.len() as f32 * 0.5f32) as usize);
                let p1 = p0 + (unwrapped0.len() as f32 * 0.5f32) as usize;

                // Crossover
                unwrapped0[p0..p1].clone_from_slice(&unwrapped1[p0..p1]);
                *network = unwrapped0;
            }
        });
    }

    fn mutation(&mut self, next_networks: &mut Vec<Vec<f32>>, elite_amount: usize) {
        let normal = Normal::new(0f64, (4f64 / next_networks[0].len() as f64).sqrt()).unwrap();

        // Mutate
        next_networks.par_iter_mut().for_each(|network| {
            let mut rng = thread_rng();
            for i in 0..network.len() {
                if rng.gen::<f32>() < self.mutation_probability {
                    network[i] = normal.sample(&mut rng) as f32
                }
            }
        });

        // Replace networks
        self.networks[elite_amount..].par_iter_mut().enumerate().for_each(|(i, network)| {
            network.copy(&next_networks[i]);
        });
        /*for i in elite_amount..self.networks.len() {
            self.networks[i].copy(&next_networks[i - elite_amount]);
        }*/
    }
}