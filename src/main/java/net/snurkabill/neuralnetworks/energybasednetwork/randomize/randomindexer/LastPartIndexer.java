package net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer;

public class LastPartIndexer extends RandomIndexer {

    public LastPartIndexer(int sizeOfVector, int numOfRandomElements, long seed) {
        super(sizeOfVector, numOfRandomElements, seed);
    }

    @Override
    public boolean[] generateIndexes() {
        boolean[] indexes = new boolean[sizeOfVector];
        int treshold = indexes.length - this.getNumberOfUnknown();
        for (int i = 0; i < indexes.length; i++) {
            indexes[i] = i >= treshold;
        }
        return indexes;
    }
}
