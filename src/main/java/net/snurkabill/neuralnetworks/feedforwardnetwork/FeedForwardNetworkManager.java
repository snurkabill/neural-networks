package net.snurkabill.neuralnetworks.feedforwardnetwork;

import net.snurkabill.neuralnetworks.data.Database;
import static net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkManager.LOGGER;
import net.snurkabill.neuralnetworks.feedforwardnetwork.target.SeparableTargetValues;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;

public class FeedForwardNetworkManager {
	
	public static final Logger LOGGER = LoggerFactory.getLogger("FFNN Manager");
	
	private final FastFeedForwardNeuralNetwork network;
	// TODO: general evaluator
	private final SeparableTargetValues targetMaker;
	private final Database database;
	
	// TODO: private final BackPropagateHeuristic heuristic;
	
	public FeedForwardNetworkManager(FastFeedForwardNeuralNetwork network, Database database) {
		this.network = network;
		this.targetMaker = new SeparableTargetValues(network.getTransferFunction());
		this.database = database;
	}
	
	public void pretrainInputNeurons(int percentageOfDataset) {
		LOGGER.info("Pretraining input neurons");
		if(percentageOfDataset > 100) {
			LOGGER.warn("More than 100% is pointless, lowering to 100%");
			percentageOfDataset = 100;
		}
		if(percentageOfDataset <= 0) {
			LOGGER.warn("Less than or equals to 0% is pointless, skipping");
			return;
		}
		int pretrainedPatterns = (database.getTrainingsetSize() / 100) * percentageOfDataset;
		double[] batch = new double [pretrainedPatterns];
		for (int i = 0; i < this.network.getSizeOfInputVector(); i++) {
			for (int j = 0; j < pretrainedPatterns; j++) {
				batch[j] = database.randomizedGetTrainingData().data[i];
			}
			network.updateInputModifier(i, batch);
		}
	}
	
	public SupervisedTestResults testNetworkOnWholeTestingDataset() {
		long testingStarted = System.currentTimeMillis();
		SupervisedTestResults results = new SupervisedTestResults();
		double[] targetValues = this.initializeTargetArray();
		double globalError = 0.0;
		int success = 0;
		int fail = 0;
		for (int i = 0; i < database.getNumberOfClasses(); i++) {
			targetMaker.getTargetValues(i, i, targetValues);
			for (int j = 0; j < database.getSizeOfTestingDataset(i); j++) {
				LOGGER.debug("testing {} iteration {}", j, database.getSizeOfTestingDataset(i));
				network.feedForwardWithPretrainedInputs(database.getTestingIteratedData(i).data);
				globalError += network.getError(targetValues);
				
				//TODO : general evaluation 
				if(network.getFirstHighestValueIndex() == i) {
					success++;
				} else fail++;
			}
		}
		int all = (success + fail);
		results.setNumOfTestedItems(database.getTestsetSize());
		results.setGlobalError(globalError);
		results.setSuccessPercentage((success * 100.0) / all);
		long testingFinished = System.currentTimeMillis();
		double sec = ((testingFinished - testingStarted) / 1000.0);
		LOGGER.info("Testing: {} samples took {} seconds, {} samples/sec", all, sec, all/sec);
		return results;
	}
	
	public void trainNetwork(int numOfIterations) {
		if(numOfIterations < database.getNumberOfClasses()) {
			LOGGER.warn("Count of iterations for training([{}]) is smaller than number of classes of division([{}]). "
					+ "Uneffective training possible: setting numOfIterations to numberOfClasses", numOfIterations, 
					database.getNumberOfClasses());
			numOfIterations = database.getNumberOfClasses();
		}
		if(numOfIterations <= 0) {
			LOGGER.warn("Number Of Iterations for training is set to [{}], skipping", numOfIterations);
			return;
		}
		long trainingStarted = System.currentTimeMillis();
		double[] targetValues = this.initializeTargetArray();
		for (int i = 1; i < numOfIterations + 1; i++) {
			int newIndex = i % database.getNumberOfClasses();
			network.feedForwardWithPretrainedInputs(database.getTrainingIteratedData(newIndex).data);
			network.backPropagation(targetMaker.getTargetValues(newIndex, 
					((i - 1) % database.getNumberOfClasses()), targetValues));
		}
		long trainingEnded = System.currentTimeMillis();
		double sec = ((trainingEnded - trainingStarted) / 1000.0);
		LOGGER.info("Training: {} samples took {} seconds, {} samples/sec", numOfIterations, sec, numOfIterations/sec);
	}
	
	private double[] initializeTargetArray() {
		double[] targetValues = new double [network.getSizeOfOutputVector()];
		for (int i = 0; i < network.getSizeOfOutputVector(); i++) {
			targetValues[i] = network.getTransferFunction().getLowLimit();
		}
		return targetValues;
	}
}
