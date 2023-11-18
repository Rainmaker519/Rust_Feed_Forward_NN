/* 

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
*/