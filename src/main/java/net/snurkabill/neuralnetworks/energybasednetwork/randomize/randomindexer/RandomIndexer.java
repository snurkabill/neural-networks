package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer;

import java.util.Random;

public abstract class RandomIndexer implements RandomDimensionsIndexer {

    protected Random random;
    protected int sizeOfVector;
    protected int numOfRandomElements;
    protected double ratio;

    public RandomIndexer(int sizeOfVector, int numOfRandomElements, long seed) {
        this.random = new Random(seed);
        this.sizeOfVector = sizeOfVector;
        this.numOfRandomElements = numOfRandomElements;
        this.ratio = (double)numOfRandomElements/(double)sizeOfVector;
    }

    public Random getRandom() {
        return random;
    }

    public int getSizeOfVector() {
        return sizeOfVector;
    }

    public int getNumOfRandomElements() {
        return numOfRandomElements;
    }

    public double getRatio() {
        return ratio;
    }

    @Override
    public int getNumberOfUnknown() {
        return this.numOfRandomElements;
    }
}
