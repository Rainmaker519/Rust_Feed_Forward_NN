use std::mem;

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
    incoming_edges: Vec<Edge>, //Vec of indices in the next layer this Node connects to
    outgoing_edges: Vec<Edge>, //Vec of indices in the previous layer which connect to this Node
    activation_type: ActivationType,
}
impl Node {
    fn new (layer: usize, number: usize, node_type: NodeType, activation_type: ActivationType) -> Self {
        Node {
            layer: layer,
            number: number,
            node_type: node_type,
            pre_activation_value: 0.0,
            post_activation_value: 0.0,
            incoming_edges: Vec::new(),
            outgoing_edges: Vec::new(),
            delta: 0.0,
            bias: 0.0,
            bias_delta: 0.0,
            activation_type: activation_type,
        }
    }

    fn add_edge_outgoing (&mut self, location_to: (usize,usize)) -> () {
        self.outgoing_edges.push(Edge::new((self.layer,self.number),location_to));
    }
    fn add_edge_incoming (&mut self, location_from: (usize,usize)) -> () {
        self.incoming_edges.push(Edge::new(location_from,(self.layer,self.number)));
    }
}

#[derive(Debug)]
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

    fn propogate_backward(delta: f64) -> () {
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
    layers: Vec<Vec<Node>>,
    number_inputs: usize,
}
impl NeuralNetwork {
    fn new (input_layer_size: usize, hidden_layer_sizes: Vec<usize>, output_layer_size: usize, loss_function: LossFunction) -> Self {
        let mut nn = NeuralNetwork {
            loss_function: loss_function,
            layers: Vec::new(),
            number_weights: 0,
            number_inputs: input_layer_size,
        };
        let num_layers: usize = hidden_layer_sizes.len() + 2;
        for layer in 0..num_layers {
            println!("Making layer {layer}: ");
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

            let mut new_layer: Vec<Node> = Vec::new();
            for j in 0..layer_size {
                println!("  Pushing node {j} to layer {layer}.");
                new_layer.push(Node::new(layer, j, node_type.clone(), activation_type.clone()));
            }

            nn.layers.push(new_layer);

            //till need to connect all the nodes created together
            //for the feed forward version of the network.
        }
        return nn
    }

    fn get_node_ref(& self, node_location: (usize,usize)) -> & Node {
        let (layer_num,node_num) = node_location;
        & self.layers[layer_num][node_num]
    }
    fn get_mut_node_ref(&mut self, node_location: (usize,usize)) -> &mut Node {
        let (layer_num,node_num) = node_location;
        &mut self.layers[layer_num][node_num]
    }

    fn get_number_inputs(& self) -> usize {
        self.number_inputs
    }
}

fn main() {
    let nn = NeuralNetwork::new(3, vec!(4,5,6), 2, LossFunction::L1);
    println!("{:?}",nn.get_node_ref((0,1)));
}







