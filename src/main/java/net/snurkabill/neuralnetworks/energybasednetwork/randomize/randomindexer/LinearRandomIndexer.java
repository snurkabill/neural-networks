package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer;

import java.util.Random;

public class LinearRandomIndexer extends RandomIndexer {

    public LinearRandomIndexer(int sizeOfVector, int numOfRandomElements, long seed) {
        super(sizeOfVector, numOfRandomElements, seed);
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
