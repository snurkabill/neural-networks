package net.snurkabill.neuralnetworks.feedforwardnetwork.manager;

import java.util.ArrayList;
import java.util.List;

import net.snurkabill.neuralnetworks.feedforwardnetwork.core.FeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class FeedForwardNetworkOnlineManager extends FeedForwardNetworkManager {

    /**
     * it is pointless to have online manager ... yet
     */

	private int lastTrained =  -1;
	private TrainingMode trainingMode;

	private int MAX_PRETRAINING_BATCH;
	private int actualBatch = 0;
	private double [][] inputBatch;
	
	public FeedForwardNetworkOnlineManager(List<FeedForwardNeuralNetwork> networks, boolean enableThreads) {
		super(networks, enableThreads);
		trainingMode = TrainingMode.SEQUENTIAL_TRAINING;
	}
	
	public FeedForwardNetworkOnlineManager(List<FeedForwardNeuralNetwork> networks, 
			int maxBatchSamplesOfPretraining, boolean enableThreads) {
		super(networks, enableThreads);
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
			/*if(actualBatch == MAX_PRETRAINING_BATCH) {
				trainingMode = TrainingMode.FULL_TRAINING;
				LOGGER.info("PRE_TRAINING mode stopped. {} is on", trainingMode);
				for (int i = 0; i < networks.size(); i++) {
					for (int j = 0; j < networks.get(0).getSizeOfInputVector(); j++) {
						networks.get(i).updateInputModifier(j, inputBatch[j]);
					}
				}
				inputBatch = null;
			} else {
				for (int i = 0; i < inputVector.length; i++) {
					inputBatch[i][actualBatch] = inputVector[i];
				}
				actualBatch++;
			}*/
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
		if(trainingMode == TrainingMode.PRE_TRAINING && inputBatch == null) {
			LOGGER.info("OnlineManager already pretrained networks. It is not possible to pretrain them again");
		}
		LOGGER.debug("Changing training mode to {}", trainingMode);
		this.trainingMode = trainingMode;
	}
	
	public static enum TrainingMode {
		FULL_TRAINING, // train on all inputs
		SEQUENTIAL_TRAINING, // train sequentially on classes of division {1, 2, 3 ....}
		@Deprecated
        PRE_TRAINING, // store batch for pretraining neurons
		NON_TRAINING; // back propagation off
	}
}
