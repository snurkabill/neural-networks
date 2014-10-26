package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomgenerator;

public class ZeroValueDimensionGenerator extends RandomBinaryDimensionsGenerator {

    public ZeroValueDimensionGenerator(int sizeOfVector, long seed) {
        super(sizeOfVector, seed);
    }

    @Override
    public double[] generateVector(double[] originalVector, boolean[] randomIndexes) {
        for (int i = 0; i < workingCopy.length; i++) {
            if (randomIndexes[i]) {
                workingCopy[i] = 0; // 2 because of binary state;
            } else {
                workingCopy[i] = originalVector[i];
            }
        }
        return workingCopy;
    }
}
