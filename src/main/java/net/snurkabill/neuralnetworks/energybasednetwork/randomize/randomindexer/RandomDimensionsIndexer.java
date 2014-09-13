package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer;

public interface RandomDimensionsIndexer {

    /**
     * index[i] == true - RANDOM VALUE
     * @return
     */
    public boolean [] generateIndexes();
}
