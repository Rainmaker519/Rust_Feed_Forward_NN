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
struct Edge<'b> {
    weight: f64,
    weight_delta: f64,
    node_from: &'b Node<'b>,
    node_to: &'b Node<'b>,
}
impl Edge <'_> {
    fn new<'b> (&mut self, layer_from: usize, number_from: usize, layer_to: usize, number_to: usize) -> () {
        Edge {
            weight = 0.0,
            weight_delta = 0.0,
            node_from
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
struct NeuralNetwork<'a> {
    loss_function: LossFunction,
    number_weights: usize,
    layers: Vec<Vec<Node<'a>>>,
}
impl NeuralNetwork <'_> {
    fn new<'a> (input_layer_size: usize, hidden_layer_sizes: Vec<usize>, output_layer_size: usize, loss_function: LossFunction) -> Self {
        let mut nn = NeuralNetwork {
            loss_function: loss_function,
            layers: Vec::new(),
            number_weights: 0,
        };
        let num_layers: usize = hidden_layer_sizes.len() + 2;
        for layer in 0..num_layers {
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
                new_layer.push(Node::new(layer, j, node_type.clone(), activation_type.clone()));
            }
        }
        return nn
    }
}

fn main() {
    let nn = NeuralNetwork::new(3, vec!(4,5,6), 2, LossFunction::L1);
    println!("{:?}",nn);
}







