package net.snurkabill.neuralnetworks.feedforwardnetwork;

import java.util.ArrayList;
import java.util.List;
import net.snurkabill.neuralnetworks.data.Database;
import static net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager.LOGGER;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;

public class FeedForwardNetworkOfflineManager extends FeedForwardNetworkManager {
	
	public static final Logger LOGGER = LoggerFactory.getLogger("FFNN Manager");
	
	// TODO: general evaluator
	private final Database database;
	
	public FeedForwardNetworkOfflineManager(List<FeedForwardNeuralNetwork> networks, Database database) {
		super(networks);
		this.database = database;
		LOGGER.info("Offline manager created with {} networks", networks.size());
	}

	public void pretrainInputNeurons(int percentageOfDataset) {
		LOGGER.info("Pretraining input neurons on {} percentages of dataset", percentageOfDataset);
		if(percentageOfDataset > 100) {
			throw new IllegalArgumentException("PercentageOfDataset makes no sense" + percentageOfDataset);
		}
		for (FeedForwardNeuralNetwork network : networks) {
			int pretrainedPatterns = (database.getTrainingsetSize() / 100) * percentageOfDataset;
			double[] batch = new double [pretrainedPatterns];
			for (int i = 0; i < network.getSizeOfInputVector(); i++) {
				for (int j = 0; j < pretrainedPatterns; j++) {
					batch[j] = database.getRandomizedTrainingData().data[i];
				}
				network.updateInputModifier(i, batch);
			}	
		}
	}
	
	public List<BasicTestResults> testNetworkOnWholeTestingDataset() {
		List<BasicTestResults> results = new ArrayList<>();
		for (int i = 0; i < networks.size(); i++) {
			results.add(new SupervisedTestResults());
		}
		for (int network = 0; network < networks.size(); network++) {
			long testingStarted = System.currentTimeMillis();
			double[] targetValues = this.initializeTargetArray(network);
			double globalError = 0.0;
			int success = 0;
			int fail = 0;
			for (int i = 0; i < database.getNumberOfClasses(); i++) {
				targetMaker.get(network).getTargetValues(i, i, targetValues);
				for (int j = 0; j < database.getSizeOfTestingDataset(i); j++) {
					networks.get(network).feedForwardWithPretrainedInputs(database.getTestingIteratedData(i).data);
					globalError += networks.get(network).getError(targetValues);

					//TODO : general evaluation 
					if(networks.get(network).getFirstHighestValueIndex() == i) {
						success++;
					} else fail++;
				}
			}
			long testingFinished = System.currentTimeMillis();
			int all = (success + fail);
			results.get(network).setNumOfTestedItems(database.getTestsetSize());
			results.get(network).setGlobalError(globalError);
			((SupervisedTestResults) results.get(network)).setSuccessPercentage((success * 100.0) / all);
			results.get(network).setMilliseconds(testingFinished - testingStarted);
			double sec = ((testingFinished - testingStarted) / 1000.0);
			LOGGER.info("Testing {}th network: {} samples took {} seconds, {} samples/sec", network, all, sec, all/sec);
		}	
		return results;
	}
	
	public void trainNetwork(int numOfIterations) {
		if(numOfIterations <= 0) {
			throw new IllegalArgumentException("Number of iterations [" + numOfIterations + "] can't be negative");
		}
		if(numOfIterations < database.getNumberOfClasses()) {
			LOGGER.warn("Count of iterations for training([{}]) is smaller than number of classes of division([{}]). "
					+ "Uneffective training possible: setting numOfIterations to numberOfClasses", numOfIterations, 
					database.getNumberOfClasses());
			numOfIterations = database.getNumberOfClasses();
		}
		for (int network = 0; network < networks.size(); network++) {
			long trainingStarted = System.currentTimeMillis();
			double[] targetValues = this.initializeTargetArray(network);
			for (int i = 1; i < numOfIterations + 1; i++) {
				int newIndex = i % database.getNumberOfClasses();
				networks.get(network).feedForwardWithPretrainedInputs(database.getTrainingIteratedData(newIndex).data);
				networks.get(network).backPropagation(targetMaker.get(network).getTargetValues(newIndex, 
						((i - 1) % database.getNumberOfClasses()), targetValues));
			}
			long trainingEnded = System.currentTimeMillis();
			double sec = ((trainingEnded - trainingStarted) / 1000.0);
			LOGGER.info("Training: {} samples took {} seconds, {} samples/sec", numOfIterations, sec,
					numOfIterations/sec);	
		}
	}
}
