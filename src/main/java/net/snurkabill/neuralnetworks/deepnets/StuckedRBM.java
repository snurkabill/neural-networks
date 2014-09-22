package net.snurkabill.neuralnetworks.deepnets;

import net.snurkabill.neuralnetworks.energybasednetwork.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardableNetwork;

import java.util.List;

public class StuckedRBM extends FeedForwardableNetwork {

    private final List<RestrictedBoltzmannMachine> levels;
    private final int indexOfLastLayer;
    private final int sizeOfOutputVector;
    private final int sizeOfInputVector;

    public StuckedRBM(List<RestrictedBoltzmannMachine> levels, String name) {
        super(name, "SuckedRBM");
        /*if(levels.size() == 1) {
            throw new IllegalArgumentException("StuckedRBM does not support one level RMBs, use one RBM instead for "
                    + "same effect");
        } else */if(levels.size() <= 0) {
            throw new IllegalArgumentException("Wrong number of levels");
        }
        this.levels = levels;
        this.indexOfLastLayer = levels.size() - 1;
        this.sizeOfInputVector = levels.get(0).getSizeOfVisibleVector();
        this.sizeOfOutputVector = levels.get(indexOfLastLayer).getSizeOfHiddenVector();
        LOGGER.info("Stucked RBMs");
    }

    public void trainNetworkLevel(double[] inputVector, int level) {
        if(level > this.indexOfLastLayer) {
            throw new IllegalArgumentException("Level (" + level + ") is too high");
        }
        if(level < 0) {
            throw new IllegalArgumentException("Level can't be negative");
        }
        checkInputVectorSize(inputVector);
        LOGGER.debug("Training inner network level {}", level);
        innerFeedForward(inputVector, level);
        levels.get(level).trainMachine();
    }

    public double[] feedForward(double[] inputVector) {
        checkInputVectorSize(inputVector);
        LOGGER.debug("Feed forwarding");
        innerFeedForward(inputVector, indexOfLastLayer);
        levels.get(this.indexOfLastLayer).activateHiddenNeurons();
        return levels.get(this.indexOfLastLayer).getHiddenNeurons();
    }

    private void innerFeedForward(double[] inputVector, int level) {
        LOGGER.trace("Inner feed forwarding to level: {}", level);
        levels.get(0).setVisibleNeurons(inputVector);
        for (int i = 0; i < level; i++) {
            levels.get(i + 1).setVisibleNeurons(levels.get(i).activateHiddenNeurons());
        }
    }

    private void checkInputVectorSize(double[] inputVector) {
        if(inputVector.length != this.sizeOfInputVector) {
            throw new IllegalArgumentException("Different size of inputVector");
        }
    }

    public int getNumOfLevels() {
        return this.levels.size();
    }
}
