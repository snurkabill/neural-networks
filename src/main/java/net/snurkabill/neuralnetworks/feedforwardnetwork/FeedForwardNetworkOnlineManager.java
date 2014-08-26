package net.snurkabill.neuralnetworks.feedforwardnetwork;

import java.util.ArrayList;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FeedForwardNetworkOnlineManager extends FeedForwardNetworkManager {
	
	public static final Logger LOGGER = LoggerFactory.getLogger("Online Manager");
	
	private int lastTrained =  -1;
	private TrainingMode trainingMode;

	private int MAX_PRETRAINING_BATCH;
	private int actualBatch = 0;
	private double [][] inputBatch;
	
	public FeedForwardNetworkOnlineManager(List<FeedForwardNeuralNetwork> networks) {
		super(networks);
		trainingMode = TrainingMode.SEQUENTIAL_TRAINING;
	}
	
	public FeedForwardNetworkOnlineManager(List<FeedForwardNeuralNetwork> networks, int maxBatchSamplesOfPretraining) {
		super(networks);
		trainingMode = TrainingMode.PRE_TRAINING;
		MAX_PRETRAINING_BATCH = maxBatchSamplesOfPretraining;
		inputBatch = new double[networks.get(0).getSizeOfInputVector()][MAX_PRETRAINING_BATCH];
	}
	
	public List<Integer> feedForwardNetwork(double [] inputVector) {
		LOGGER.debug("feed forwarding");
		checkInput(inputVector);	
		for (int network = 0; network < networks.size(); network++) {
			networks.get(network).feedForward(inputVector);
		}
		List<Integer> tmp = new ArrayList<>();
		for (int network = 0; network < networks.size(); network++) {
			tmp.add(networks.get(network).getFirstHighestValueIndex());
		}
		return tmp;
	}
	
	private void backPropagation(int expectedOutputClassIndex) {
		lastTrained = expectedOutputClassIndex;
		for (int network = 0; network < networks.size(); network++) {
			double[] targetValues = this.initializeTargetArray(network);
			networks.get(network).backPropagation(targetMaker.get(network).getTargetValues(expectedOutputClassIndex, 
					expectedOutputClassIndex, targetValues));
		}
	}
	
	public List<Integer> insertInNetwork(double [] inputVector, int expectedOutputClassIndex) {
		List<Integer> results = feedForwardNetwork(inputVector);
		if(trainingMode == TrainingMode.FULL_TRAINING) {
			LOGGER.debug("FULL_TRAINING {}", expectedOutputClassIndex);
			backPropagation(expectedOutputClassIndex);
		} else if(trainingMode == TrainingMode.SEQUENTIAL_TRAINING) {
			LOGGER.debug("SEQUENTIAL_TRAINING {}", expectedOutputClassIndex);
			if(expectedOutputClassIndex == nextToTrain()) {
				backPropagation(expectedOutputClassIndex);
			}
		} else if(trainingMode == TrainingMode.PRE_TRAINING) {
			LOGGER.debug("PRE_TAINING {}", expectedOutputClassIndex);
			if(actualBatch == MAX_PRETRAINING_BATCH) {
				trainingMode = TrainingMode.SEQUENTIAL_TRAINING;
				LOGGER.info("PRE_TRAINING mode stopped. {} is on", trainingMode);
				for (int i = 0; i < networks.size(); i++) {
					for (int j = 0; j < networks.get(0).getSizeOfInputVector(); j++) {
						networks.get(i).updateInputModifier(j, inputBatch[j]);
					}
				}
			} else {
				for (int i = 0; i < inputVector.length; i++) {
					inputBatch[i][actualBatch] = inputVector[i];
				}
				actualBatch++;
			}
		}
		return results;
	}
	
	private void checkInput(double [] inputVector) {
		if(inputVector.length != this.sizeOfInputVector) {
			throw new IllegalArgumentException("Size of input vector {" + inputVector.length + "} is different than "
					+ "inputSize of neural networks");
		}
	}
	
	public int nextToTrain() {
		if(lastTrained == this.sizeOfOutputVector - 1) {
			return 0;
		}
		return lastTrained + 1;
	}
	
	public void changeTrainingMode(TrainingMode trainingMode) {
		LOGGER.debug("Changing training mode to {}", trainingMode);
		this.trainingMode = trainingMode;
	}
	
	public static enum TrainingMode {
		FULL_TRAINING, // train on all inputs
		SEQUENTIAL_TRAINING, // train sequential on classes of division {1, 2, 3 ....}
		PRE_TRAINING, // store batch for pretraining neurons
		NON_TRAINING; // back propagation off
	}
}
