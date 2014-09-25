package net.snurkabill.neuralnetworks.deepnets;

import net.snurkabill.neuralnetworks.energybasednetwork.RestrictedBoltzmannMachine;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

/**
 * NOT FINISHED
 */

/**
 * TODO : make DBN abstract and create binary and real valued DBN. 
 */
@Deprecated
public class ReconstructingDeepBeliefNeuralNetwork extends StuckedRBM {

	public static final Logger LOGGER = LoggerFactory.getLogger("DBN");
	private final List<RestrictedBoltzmannMachine> levels;
	private final int indexOfLastLayer;
	private final int indexOfLastInnerLayer;
	private final String name;
	private final int sizeOfOutputVector;
	private final int sizeOfInputVector;
	
	public ReconstructingDeepBeliefNeuralNetwork(List<RestrictedBoltzmannMachine> levels, String name) {
        super(levels, name);
		if(levels.size() == 1) {
			throw new IllegalArgumentException("DBN does not support one level RMBs, use one RBM instead for "
					+ "same effect");
		} else if(levels.size() <= 0) {
			throw new IllegalArgumentException("Wrong number of levels");
		}
		this.levels = levels;
		this.indexOfLastLayer = levels.size() - 1;
		this.indexOfLastInnerLayer = indexOfLastLayer - 1;
		this.name = name;
		this.sizeOfInputVector = levels.get(0).getSizeOfVisibleVector();
		this.sizeOfOutputVector = levels.get(indexOfLastLayer).getSizeOfVisibleVector() - 
				levels.get(indexOfLastLayer - 1).getSizeOfHiddenVector();
		LOGGER.info("Created Deep Belief Network");
	}

	public String getName() {
		return name;
	}
	
	public void trainInnerNetworkLevel(double[] inputVector, int level) {
		if(level == this.indexOfLastLayer) {
			throw new IllegalArgumentException("Level (" + level + ") can't be trained this way");
		}
		if(level < 0) {
			throw new IllegalArgumentException("Level can't be negative");
		}
		checkInputVectorSize(inputVector);
		LOGGER.debug("Training inner network level {}", level);
		innerFeedForward(inputVector, level);
		levels.get(level).trainMachine();
	}
	
	public void trainLastNetworkLayer(double[] inputVector, double[] label) {
		checkInputVectorSize(inputVector);
		int inputLength = levels.get(indexOfLastLayer - 1).getSizeOfHiddenVector() + label.length;
		if(levels.get(indexOfLastLayer).getSizeOfVisibleVector() != inputLength) {
			throw new IllegalArgumentException("Different size of inputVector. Expected size (" + 
					levels.get(indexOfLastLayer).getSizeOfVisibleVector() + ") actual size (" + 
					inputLength + ").");
		}
		LOGGER.debug("Training last network level");
		LOGGER.trace("Size of reconstructed vector: {}, size of labels {}, size of lastHidden {}", 
				levels.get(indexOfLastLayer).getSizeOfVisibleVector(), label.length, 
				levels.get(indexOfLastInnerLayer).getSizeOfHiddenVector());
		double[] reconstructed = new double[levels.get(indexOfLastLayer).getSizeOfVisibleVector()];
		double[] lastHidden = innerFeedForward(inputVector);
		LOGGER.info("{}", lastHidden);
		for (int i = 0; i < lastHidden.length; i++) {
			reconstructed[i] = lastHidden[i];
		}
		for (int i = 0; i < label.length; i++) {
			reconstructed[i + lastHidden.length] = label[i];
		}
		levels.get(indexOfLastLayer).trainMachine(reconstructed);
	}
	
	public double[] feedForward(double[] inputVector, int numOfRuns) {
		checkInputVectorSize(inputVector);
		if(numOfRuns <= 0) {
			throw new IllegalArgumentException("Wrong number of runs of DNB");
		}
		LOGGER.debug("Feed forwarding");
		double[] output = new double[this.sizeOfOutputVector];
		double[] reconstructed = new double[levels.get(indexOfLastLayer).getSizeOfVisibleVector()];
		for (int i = 0; i < this.sizeOfOutputVector; i++) {
			output[i] = 0.0;
		}
		for (int run = 0; run < numOfRuns; run++) {
			double[] tmp = innerFeedForward(inputVector);
			for (int i = 0; i < tmp.length; i++) {
				reconstructed[i] = tmp[i];
			}
			for (int i = tmp.length; i < reconstructed.length; i++) {
				reconstructed[i] = 0.0;		// set reconstructed part of vector to zeros
			}
			levels.get(indexOfLastLayer).setVisibleNeurons(reconstructed);
			levels.get(indexOfLastLayer).activateHiddenNeurons();
			reconstructed = levels.get(indexOfLastLayer).activateVisibleNeurons();
			
			for (int i = 0; i < this.sizeOfOutputVector; i++) {
				output[i] += reconstructed[i + tmp.length];
			}
		}
		for (int i = 0; i < output.length; i++) {
			output[i] /= numOfRuns;
		}
		return output;
	}
	
	private double[] innerFeedForward(double[] inputVector) {
		innerFeedForward(inputVector, indexOfLastInnerLayer);
		levels.get(this.indexOfLastInnerLayer).activateHiddenNeurons();
		return levels.get(this.indexOfLastInnerLayer).getHiddenNeurons();
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
}
