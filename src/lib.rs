use std::rc::Rc;

#[derive(Debug)]
struct Node <'c> {
    layer: usize,
    number: usize,
    node_type: NodeType,
    pre_activation_value: f64,
    post_activation_value: f64,
    delta: f64,
    bias: f64,
    bias_delta: f64,
    incoming_edges: Box<Vec< &'c Edge<'c> >>, //Vec of indices in the next layer this Node connects to
    outgoing_edges: Box<Vec< &'c Edge<'c> >>, //Vec of indices in the previous layer which connect to this Node
    activation_type: ActivationType,
}
impl Node <'_> {
    fn new<'c> (layer: usize, number: usize, node_type: NodeType, activation_type: ActivationType) -> Self {
        Node {
            layer: layer,
            number: number,
            node_type: node_type,
            pre_activation_value: 0.0,
            post_activation_value: 0.0,
            incoming_edges: Box::new(Vec::new()),
            outgoing_edges: Box::new(Vec::new()),
            delta: 0.0,
            bias: 0.0,
            bias_delta: 0.0,
            activation_type: activation_type,
        }
    }

    fn add_incoming_edge<'incoming_edge_life> (&'incoming_edge_life mut self, edge_incoming: Edge<'incoming_edge_life>) -> () {
        //check if edge exists already
        for edge in &mut * self.incoming_edges {
            match edge.node_from {
                Some(node_from_exists) => {
                    match edge_incoming.node_from {
                        Some(node_from_potential) => {
                            if node_from_exists.layer == node_from_potential.layer && node_from_exists.number == node_from_potential.number {
                                //Layer is the same!
                                match edge.node_to {
                                    Some(node_to_exists) => {
                                        match edge_incoming.node_to {
                                            Some(node_to_potential) => {
                                                if node_to_exists.layer == node_to_potential.layer && node_to_exists.number == node_to_potential.number {
                                                    //Connection already exists!!!
                                                    ()
                                                }
                                            }
                                            _ => panic!(),
                                        }
                                    }
                                    _ => panic!(),
                                }
                            }
                        }
                        _ => panic!(),
                    }
                }
                _ => panic!(),
            }
        }
        //if make it to here, then edge should be added
        match Box::froself.incoming_edges {
            Some(x) => {

            }
            _ => {

            }
        }

        }
        ()
        //ahaha
    }

    fn add_outgoing_edge<'outgoing_edge_life> (&'outgoing_edge_life mut self, edge_outgoing: &'outgoing_edge_life Edge<'outgoing_edge_life>) -> () {
        //check if edge exists already
        for edge in &mut * self.outgoing_edges {
            match edge.node_from {
                Some(node_from_exists) => {
                    match edge_outgoing.node_from {
                        Some(node_from_potential) => {
                            if node_from_exists.layer == node_from_potential.layer && node_from_exists.number == node_from_potential.number {
                                //Layer is the same!
                                match edge.node_to {
                                    Some(node_to_exists) => {
                                        match edge_outgoing.node_to {
                                            Some(node_to_potential) => {
                                                if node_to_exists.layer == node_to_potential.layer && node_to_exists.number == node_to_potential.number {
                                                    //Connection already exists!!!
                                                    ()
                                                }
                                                else {
                                                    //Make the connection!
                                                    self.outgoing_edges.push(edge_outgoing);
                                                    ()
                                                }
                                            }
                                            _ => panic!(),
                                        }
                                    }
                                    _ => panic!(),
                                }
                            }
                            else {
                                //Different, so make the connection!
                                self.outgoing_edges.push(edge_outgoing);
                                ()
                            }
                        }
                        _ => panic!(),
                    }
                }
                _ => panic!(),
            }
        }
    }

    //Reset the values which need to be recalculated for each forward and backward pass.
    //Will also reset the deltas for outgoing edges.
    fn reset<'reset_life> (&'reset_life mut self) -> () {
        self.pre_activation_value = 0.0;
        self.post_activation_value = 0.0;
        self.delta = 0.0;
        self.bias_delta = 0.0;

        for i in &mut * self.outgoing_edges {
            i.weight_delta = 0.0;
        }
    }

    fn add_edge_to_node_from<'add_edge_from>(&'add_edge_from mut self, edge: &'add_edge_from Edge) -> () {
        self.outgoing_edges.push(edge);
    }
    fn add_edge_to_node_to<'add_edge_to>(&'add_edge_to mut self, edge: &'add_edge_to Edge) -> () {
        self.incoming_edges.push(edge);
    }

    fn add_edge_outgoing(layer: usize, number: usize) -> () {

    }

    //used to get all weights by iterating over all nodes
    //with an initially empty but updated vector
    //
    //returns the number of weights retrieved
    //(1 for bias plus 1 for each outgoing edge)
    //and updates the passed vector with those weights
    //starting at the given position
    fn get_weights(self, position: usize, weights: &mut Vec<f64>) -> usize {
        let mut weight_count = 0;

        //the first weight set will be the bias if
        //it is a hidden node
        if self.node_type == NodeType::HIDDEN {
            weights[position] = self.bias;
            weight_count = 1;
        }

        for edge in self.outgoing_edges {
            weights[position + weight_count] = edge.weight;
            weight_count = weight_count + 1;
        }

        return weight_count;
    }
    //works in the same way as get_weights() but for the deltas instead
    fn get_deltas(self, position: usize, deltas: &mut Vec<f64>) -> usize {
        let mut delta_count = 0;

        //the first weight set will be the bias if
        //it is a hidden node
        if self.node_type == NodeType::HIDDEN {
            deltas[position] = self.bias_delta;
            delta_count = 1;
        }

        for edge in self.outgoing_edges {
            deltas[position + delta_count] = edge.weight_delta;
            delta_count = delta_count + 1;
        }

        return delta_count;
    }

    fn set_weights(&mut self, position: usize, weights: Vec<f64>) -> usize {
        let mut weight_count: usize = 0;

        if self.node_type == NodeType::HIDDEN {
            self.bias = weights[position];
            weight_count =  1;
        }

        for edge in self.outgoing_edges {
            edge.weight = weights[position + weight_count];
            weight_count = weight_count + 1;
        }
        return weight_count
    }
    //Implement the following for P1-1
    fn propogate_forward() -> () {

    }
    fn propogate_backward() -> () {

    }

    //Implement the following for P1-3
    fn apply_linear() -> () {

    }
    fn apply_sigmoid() -> (){

    }
    fn apply_tanh() -> (){

    }
    fn apply_softmax() -> (){

    }
    //
}





















#[derive(Debug)]
struct Edge <'a> {
    weight: f64,
    weight_delta: f64,
    node_from: Option<&'a Node<'a>>,
    node_to: Option<&'a Node<'a>>,
}
impl Edge <'_> {
    fn new (node1: Node, node2: Node) -> *mut Self {
        let edge = Edge {
            weight: 0.0,
            weight_delta: 0.0,
            node_from: None,
            node_to: None,
        };
        Box::into_raw(Box::new(edge))
    }

    fn set_node_to(&mut self, node: & Node) -> Edge {
        self.node_to = Option::Some(node);
        * self
    }
    fn set_node_from(&mut self, node: & Node) -> Edge {
        self.node_from = Option::Some(node);
        * self
    }

    fn propogate_backward(delta: f64) -> () {
        //implement for P1-2
    }
}














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