package net.snurkabill.neuralnetworks.neuralnetwork.deep;

import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class StuckedRBM implements DeepNetwork {

    private final Logger LOGGER = LoggerFactory.getLogger("StuckedRBM");

    private final String name;
    private final List<RestrictedBoltzmannMachine> levels;
    private final int indexOfLastLayer;
    private final int sizeOfOutputVector;
    private final int sizeOfInputVector;

    public StuckedRBM(List<RestrictedBoltzmannMachine> levels, String name) {
        String tmpName = "StuckedRBM " + levels.get(0).getSizeOfInputVector();
        for (RestrictedBoltzmannMachine level : levels) {
            tmpName = tmpName.concat("->" + level.getSizeOfOutputVector());
        }
        this.name = name;
        if(levels.size() == 1) {
            throw new IllegalArgumentException("StuckedRBM does not support one level of RMBs, use one RBM instead for "
                    + "same effect");
        } else if(levels.size() <= 0) {
            throw new IllegalArgumentException("Wrong number of levels");
        }
        this.levels = levels;
        this.indexOfLastLayer = levels.size() - 1;
        this.sizeOfInputVector = levels.get(0).getSizeOfVisibleVector();
        this.sizeOfOutputVector = levels.get(indexOfLastLayer).getSizeOfHiddenVector();
        LOGGER.info("{} created!", name);
    }

    public void trainNetworkLevel(double[] inputVector, int level) {
        if(level > this.indexOfLastLayer) {
            throw new IllegalArgumentException("Level (" + level + ") is too high");
        }
        if(level < 0) {
            throw new IllegalArgumentException("Level can't be negative");
        }
        LOGGER.debug("Training {}th level.", level);
        checkInputVectorSize(inputVector);
        innerFeedForward(inputVector, level);
        levels.get(level).trainMachine();
    }

    public double[] feedForward(double[] inputVector) {
        LOGGER.debug("Feed forwarding");
        checkInputVectorSize(inputVector);
        innerFeedForward(inputVector, indexOfLastLayer);
        return levels.get(this.indexOfLastLayer).activateHiddenNeurons();
    }

    private void innerFeedForward(double[] inputVector, int level) {
        LOGGER.trace("InnerFeedForward to level {}", level);
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

    public void calculateNetwork(double[] inputVector) {
        feedForward(inputVector);
    }

    public int getSizeOfOutputVector() {
        return sizeOfOutputVector;
    }

    public int getSizeOfInputVector() {
        return sizeOfInputVector;
    }

    public double[] getOutputValues() {
        return levels.get(indexOfLastLayer).getHiddenNeurons();
    }

    public void getOutputValues(double[] outputValues) {
        double[] tmp = levels.get(indexOfLastLayer).getHiddenNeurons();
        System.arraycopy(tmp, 0, outputValues, 0, outputValues.length);
    }
}
