package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer;

import java.util.Random;

public class LinearRandomIndexer implements RandomDimensionsIndexer {

    private Random random;
    private int sizeOfVector;
    private int numOfRandomElements;
    private double ratio;

    public LinearRandomIndexer(int sizeOfVector, int numOfRandomElements, long seed) {
        this.random = new Random(seed);
        this.sizeOfVector = sizeOfVector;
        this.numOfRandomElements = numOfRandomElements;
        this.ratio = (double)numOfRandomElements/(double)sizeOfVector;
    }

    @Override
    public boolean[] generateIndexes() {
        boolean [] indexes = new boolean [sizeOfVector];
        int randomRemaining = numOfRandomElements;
        for (; ;) {
            for (int i = 0; i < sizeOfVector; i++) {
                if(indexes[i] == false) {
                    if(random.nextDouble() <= ratio) {
                        randomRemaining--;
                        indexes[i] = true;
                    }
                }
                if(randomRemaining == 0) {
                    return indexes;
                }
            }
        }
    }
}
