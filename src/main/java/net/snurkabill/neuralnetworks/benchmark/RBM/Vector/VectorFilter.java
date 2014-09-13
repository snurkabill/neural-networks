package net.snurkabill.neuralnetworks.benchmark.RBM.Vector;

import java.util.List;

public interface VectorFilter {

    public List<Tuple> filter(List<Tuple> tuples);
}
