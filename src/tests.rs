fn main() {
    let mut node = Node::new(0.96f64,ActivationFunction::Sigmoid);
    println!("Should print 0.960 after this:");
    println!("{:.3}",node.value);

    let edge: Edge = Edge::new(1, 4);
    println!("Should print 4 after this:");
    println!("{:.2}",edge.node_index);

    node.add_edge_forward(1, 3);
    println!("Adding forward connection test:");
    println!("{:?}",node.connections_forward);
}