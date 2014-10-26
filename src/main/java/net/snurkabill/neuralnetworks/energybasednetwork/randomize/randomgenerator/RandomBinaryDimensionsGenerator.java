package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomgenerator;

import java.util.Random;

public class RandomBinaryDimensionsGenerator {

    protected final double[] workingCopy;
    protected final Random random;

    public RandomBinaryDimensionsGenerator(int sizeOfVector, long seed) {
        this.workingCopy = new double[sizeOfVector];
        this.random = new Random(seed);
    }

    public double[] generateVector(double[] originalVector, boolean[] randomIndexes) {
        for (int i = 0; i < workingCopy.length; i++) {
            if (randomIndexes[i]) {
                workingCopy[i] = random.nextInt(2); // 2 because of binary state;
            } else {
                workingCopy[i] = originalVector[i];
            }
        }
        return workingCopy;
    }
}
