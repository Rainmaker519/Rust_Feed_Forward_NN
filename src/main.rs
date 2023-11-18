#![allow(dead_code)]
#![allow(unused_variables)]

use std::cell::RefCell;
//use std::{mem, cell::{RefCell, Ref}, borrow::BorrowMut};

/// ENUM SECTION ------------------------------------------------
#[derive(PartialEq,Eq,Debug,Clone)]
enum ActivationType {
    SIGMOID,
    TANH,
    SOFTMAX,
    LINEAR,
}
#[derive(PartialEq,Eq,Debug,Clone)]
enum NodeType {
    INPUT,
    OUTPUT,
    HIDDEN,
}
#[derive(PartialEq,Eq,Debug,Clone)]
enum LossFunction {
    L1,
    L2,
}
/// END ENUM SECTION ------------------------------------------------
/// 
/// 

#[derive(Debug)]
struct Node {
    layer: usize,
    number: usize,
    node_type: NodeType,
    pre_activation_value: f64,
    post_activation_value: f64,
    delta: f64,
    bias: f64,
    bias_delta: f64,
    incoming_edges: Vec<Edge>,
    outgoing_edges: Vec<Edge>,
    activation_type: ActivationType,
}
impl Node {
    fn new (layer: usize, number: usize, node_type: NodeType, activation_type: ActivationType) -> Self {
        Node {
            layer,
            number,
            node_type,
            pre_activation_value: 0.0,
            post_activation_value: 0.0,
            incoming_edges: Vec::new(),
            outgoing_edges: Vec::new(),
            delta: 0.0,
            bias: 0.0,
            bias_delta: 0.0,
            activation_type,
        }
    }

    fn add_edge_outgoing (&mut self, location_to: (usize,usize)) -> () {
        self.outgoing_edges.push(Edge::new((self.layer,self.number),location_to));
    }
    fn add_edge_incoming (&mut self, location_from: (usize,usize)) -> () {
        self.incoming_edges.push(Edge::new(location_from,(self.layer,self.number)));
    }

    //Reset the values which need to be recalculated for each forward and backward pass.
    //Will also reset the deltas for outgoing edges.
    fn reset (&mut self) -> () {
        self.pre_activation_value = 0.0;
        self.post_activation_value = 0.0;
        self.delta = 0.0;
        self.bias_delta = 0.0;

        for i in &mut * self.outgoing_edges {
            i.weight_delta = 0.0;
        }
    }

    fn get_weights(& self, weights: &mut Vec<f64>) -> usize {
        let mut weight_count = 0;

        //the first weight set will be the bias if
        //it is a hidden node
        if self.node_type == NodeType::HIDDEN {
            weights.push(self.bias);
            weight_count = 1;
        }

        let edges_full_slice = &self.outgoing_edges;
        //println!("{:?}",edges_full_slice);

        for edge in edges_full_slice {
            //println!("{:?}",edge);
            //println!("---");
            weights.push(edge.weight);
            weight_count = weight_count + 1;
        }

        return weight_count;
    }

    //works in the same way as get_weights() but for the deltas instead
    fn get_deltas(& self, deltas: &mut Vec<f64>) -> usize {
        let mut delta_count = 0;

        //the first weight set will be the bias if
        //it is a hidden node
        if self.node_type == NodeType::HIDDEN {
            deltas.push(self.bias);
            delta_count = 1;
        }

        let edges_full_slice = &self.outgoing_edges;
        //println!("{:?}",edges_full_slice);

        for edge in edges_full_slice {
            //println!("{:?}",edge);
            //println!("---");
            deltas.push(edge.weight_delta);
            delta_count = delta_count + 1;
        }

        return delta_count;
    }
    
    fn set_weights(&mut self, position: usize, weights: &mut Vec<f64>) -> usize {
        let mut weight_count = 0;

        //the first weight set will be the bias if
        //it is a hidden node
        if self.node_type == NodeType::HIDDEN {
            self.bias = weights[position];
            weight_count = 1;
        }

        let edges_full_slice = &mut self.outgoing_edges;
        //println!("{:?}",edges_full_slice);

        for edge in edges_full_slice {
            //println!("{:?}",edge);
            //println!("---");
            edge.weight = weights[position + weight_count];
            weight_count = weight_count + 1;
        }

        return weight_count;
    }
}

#[derive(Debug,Clone)]
struct Edge {
    weight: f64,
    weight_delta: f64,
    node_from: (usize,usize),
    node_to: (usize,usize),
}
impl Edge {
    fn new(node_from_ref: (usize,usize), node_to_ref: (usize,usize)) -> Self {
        Edge {
            weight: 0.0,
            weight_delta: 0.0,
            node_from: node_from_ref,
            node_to: node_to_ref,
        }
    }

    fn propagate_backward(delta: f64) -> () {
        //implement for P1-2
    }
}
//DataSet Type

//Needs:
    //getName() -> Name of the dataset
    //getNumberInputs() -> Gives the number of inputs of the nn
    //
#[derive(Debug)]
struct NeuralNetwork {
    loss_function: LossFunction,
    number_weights: usize,
    layers: RefCell<Vec<RefCell<Vec<Node>>>>,
    number_inputs: usize,
}
impl NeuralNetwork {
    fn new (input_layer_size: usize, hidden_layer_sizes: Vec<usize>, output_layer_size: usize, loss_function: LossFunction) -> Self {
        let mut nn = NeuralNetwork {
            loss_function,
            layers: RefCell::new(Vec::new()),
            number_weights: 0,
            number_inputs: input_layer_size,
        };

        let num_layers: usize = hidden_layer_sizes.len() + 2;

        for layer in 0..num_layers {
            //println!("Making layer {layer}: ");
            let layer_size: usize;
            let node_type: NodeType;
            let activation_type: ActivationType;

            if layer == 0 {
                layer_size = input_layer_size;
                node_type = NodeType::INPUT;
                activation_type = ActivationType::LINEAR;

            }
            else if layer < num_layers - 1 {
                layer_size = hidden_layer_sizes[layer - 1];
                node_type = NodeType::HIDDEN;
                activation_type = ActivationType::TANH;

                nn.number_weights = nn.number_weights + layer_size;
            }
            else {
                layer_size = output_layer_size;
                node_type = NodeType::OUTPUT;
                activation_type = ActivationType::SIGMOID;
            }

            let mut new_layer = RefCell::new(Vec::new());
            for j in 0..layer_size {
                //println!("  Pushing node {j} to layer {layer}.");
                new_layer.get_mut().push(Node::new(layer, j, node_type.clone(), activation_type.clone()));
            }

            nn.layers.get_mut().push(new_layer);

            //if not the input layer connect all the nodes from the previous layer to this layer
            if layer != 0 {
                //already have layer
                //loop through to_node_num and from_node_num
                let mut layer_sizes: Vec<usize>;
                layer_sizes = Vec::new();
                {
                    let all_layers = nn.layers.get_mut();
                    let accessed = &mut *all_layers;
                    for layer in accessed {
                        let l = layer.get_mut();
                        layer_sizes.push(l.len());
                    }
                }
                
                {
                    let all_layers = nn.layers.get_mut();
                    let from_node_layer_option = all_layers.get_mut(layer - 1);
                    match from_node_layer_option {
                        Some(from_node_layer_ref) => {
                            let from_node_layer = from_node_layer_ref.get_mut();
                            
                            for from_node_num in 0..(from_node_layer.len()) {
                                let from_node_ref = all_layers[layer-1].get_mut().get_mut(from_node_num);
                                match from_node_ref {
                                    Some(from_node) => {
                                        for to_node_num in 0..(layer_sizes[layer]) {
                                            from_node.add_edge_outgoing((layer,to_node_num)); 
                                        }
                                        //print!("added outgoing edges to: ");
                                        //println!("{:?}",from_node);
                                    },
                                    None => panic!("no 'from node' in option"),
                                }
                            }
                        },
                        _ => panic!("no 'node layer' in option")
                    }
                }
                //println!("AAAAAAAAAAAAAAAAHHHHHHHHHHHERERERERERE~~~~!@");
                {
                    let all_layers = nn.layers.get_mut();
                    let to_node_layer_option = all_layers.get_mut(layer);
                    match to_node_layer_option {
                        Some(to_node_layer_ref) => {
                            let to_node_layer = to_node_layer_ref.get_mut();
                            for to_node_layer_num in 0..to_node_layer.len() {
                                let to_node_ref = all_layers[layer].get_mut().get_mut(to_node_layer_num);
                                match to_node_ref {
                                    Some(to_node) => {
                                        for from_node_num in 0..(layer_sizes[layer-1]) {
                                            to_node.add_edge_incoming((layer-1,from_node_num)); 
                                        }
                                        //print!("added incoming edges to: ");
                                        //println!("{:?}",to_node);
                                    },
                                    None => panic!("no 'node layer' in option (2)"),
                                } 
                            }
                        },
                        _ => panic!("no 'node layer' in option (2)")
                    }
                }
            }
        }
        return nn
    }

    fn get_node_ref(&mut self, node_location: (usize,usize)) -> & Node {
        let (layer_num,node_num) = node_location;
        let layer_option = self.layers.get_mut().get_mut(layer_num);
        match layer_option {
            Some(layer) => {
                let node_option = layer.get_mut().get(node_num);

                match node_option {
                    Some(node) => {
                        node
                    },
                    _ => panic!("get_node_ref dead 1")
                }
            },
            None => panic!("get_node_ref dead 2"),
        }
    }
    fn get_mut_node_ref(&mut self, node_location: (usize,usize)) -> &mut Node {
        let (layer_num,node_num) = node_location;
        let all_layers = self.layers.get_mut();
        let layer_option = all_layers.get_mut(layer_num);
        match layer_option {
            Some(layer) => {
                let node_option = layer.get_mut().get_mut(node_num);

                match node_option {
                    Some(node) => {
                        node
                    },
                    _ => panic!("jj spot")
                }
            },
            _ => panic!("aj spot")
        }
    }

    fn get_number_inputs(& self) -> usize {
        self.number_inputs
    }

    fn get_weights(&mut self) -> Vec<f64> {
        let mut weights: Vec<f64> = Vec::new();
        let mut position: usize = 0;

        for layer_counter in 0..self.layers.get_mut().len() {
            let current_layer_option = self.layers.get_mut().get_mut(layer_counter);
            match current_layer_option {
                Some(layer_refcell) => {
                    let layer = layer_refcell.get_mut();

                    for node_counter in 0..layer.len() {
                        let current_node_option = layer.get(node_counter);
                        match current_node_option {
                            Some(node_ref) => {
                                let n_weights = node_ref.get_weights(&mut weights);

                                position = position + n_weights;                             
                            },
                            _ => panic!("ah ah ah ah")
                        }
                    } 
                },
                _ => panic!("get weights nn die die")
            }
        }
        weights
    }

    fn set_weights(&mut self, weights: &mut Vec<f64>) -> () {
        let mut position: usize = 0;

        for layer_counter in 0..self.layers.get_mut().len() {
            let current_layer_option = self.layers.get_mut().get_mut(layer_counter);
            match current_layer_option {
                Some(layer_refcell) => {
                    let layer = layer_refcell.get_mut();

                    for node_counter in 0..layer.len() {
                        let current_node_option = layer.get_mut(node_counter);
                        match current_node_option {
                            Some(node_ref) => {
                                let n_weights = node_ref.set_weights(position, weights);
                                position = position + n_weights;                             
                            },
                            _ => panic!("a ha ha ha")
                        }
                    }
                },
                _ => panic!("set weights nn die die")
            }
        }
    }
}


fn test_set_weights() -> bool {
    false
}
fn test_get_weights_nn() -> bool {
    let mut nn = NeuralNetwork::new(3, vec!(4,2,7), 3, LossFunction::L1);
    if nn.get_weights().len() != 68 {
        false;
    }
    let mut nn = NeuralNetwork::new(3, vec!(2,2), 2, LossFunction::L1);
    if nn.get_weights().len() != 18 {
        false;
    }
    true
}

fn main() {
    println!();
    println!("---------Main Run Output Start---------\n");
    let mut nn = NeuralNetwork::new(3, vec!(4,2,7), 3, LossFunction::L1);
    //println!("{:#?}",nn.get_node_ref((1,1)).outgoing_edges);
    //println!("{:#?}",nn);

    //get_weights_test
    let get_weights_test_result = test_get_weights_nn();
    print!("GetWeightsTest: ");
    if get_weights_test_result {
        println!("Pass\n")
    }
    else {
        println!("Fail\n")
    }
    
    //The set_weights function WILL accept lists of weights over the
    //  correct size without crashing rn, need to add exception handling
    //  for that for QOL stuff.
    let mut a_weights: Vec<f64> = vec!(1.5; 68);
    

    nn.set_weights(&mut a_weights);

    let b_weights = nn.get_weights();
    println!("{:?}",b_weights);
    //set_weights_test
    
}







/*
TO DO:

DONE || 1. Implement get weight and delta functions for nn using the node implementation.

2. Implement set weight and delta functions for nn using the node implementation.

3. Check the Java implementation to figure out what else I need to make to start the assignment
for reals.


*/
