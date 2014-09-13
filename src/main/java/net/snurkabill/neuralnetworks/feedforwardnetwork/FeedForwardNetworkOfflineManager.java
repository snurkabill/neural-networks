package net.snurkabill.neuralnetworks.feedforwardnetwork;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;
import net.snurkabill.neuralnetworks.data.DataItem;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.data.LabelledItem;
import net.snurkabill.neuralnetworks.results.BasicTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;

public class FeedForwardNetworkOfflineManager extends FeedForwardNetworkManager {
	
	public static final Logger LOGGER = LoggerFactory.getLogger("FFNN Manager");
	
	// TODO: general evaluator
	private final Database database;
	
	public FeedForwardNetworkOfflineManager(List<FeedForwardNeuralNetwork> networks, Database database,
                                            boolean isMultithreadingEnabled) {
		super(networks, isMultithreadingEnabled);
		this.database = database;
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
			network.setIsPretrained(true);
		}
	}
	
	public List<BasicTestResults> testNetworkOnWholeTestingDataset() {
		List<BasicTestResults> results = new ArrayList<>();
		for (int network = 0; network < networks.size(); network++) {
			long testingStarted = System.currentTimeMillis();
			double[] targetValues = this.initializeTargetArray(network);
			int[] successValuesCounter = new int[networks.get(network).getSizeOfOutputVector()];
			double globalError = 0.0;
			int success = 0;
			int fail = 0;
			for (int _class = 0; _class < database.getNumberOfClasses(); _class++) {
				targetMaker.get(network).getTargetValues(_class, _class, targetValues);
				/*for (int j = 0; j < database.getSizeOfTestingDataset(i); j++) {
					networks.get(network).feedForward(database.getTestingIteratedData(i).data);
					globalError += networks.get(network).getError(targetValues);
					//TODO : general evaluation 
					if(networks.get(network).getFirstHighestValueIndex() == i) {
						success++;
						successValuesCounter[i]++;
					} else fail++;
				}*/
				Iterator<DataItem> testingIterator = database.getTestingIteratorOverClass(_class);
				for (;testingIterator.hasNext();) {
					networks.get(network).feedForward(testingIterator.next().data);
					globalError += networks.get(network).getError(targetValues);
					if(networks.get(network).getFirstHighestValueIndex() == _class) {
						success++;
						successValuesCounter[_class]++;
					} else fail++;
				}
			}
			long testingFinished = System.currentTimeMillis();
			int all = (success + fail);
            results.add(new SupervisedTestResults(database.getTestSetSize(), globalError,
                    testingFinished - testingStarted, (success * 100.0) / all));
			double sec = ((testingFinished - testingStarted) / 1000.0);
			LOGGER.info("Testing {} network: {} samples took {} seconds, {} samples/sec", 
					networks.get(network).getName(), all, sec, all/sec);
			for (int i = 0; i < database.getNumberOfClasses(); i++) {
				LOGGER.info("{}th class: TOTAL - {}, {}% correct samples", i, database.getSizeOfTestingDataset(i),
						(1.0 / ((double)database.getSizeOfTestingDataset(i) / (double)successValuesCounter[i])));
			}
		}	
		return results;
	}
	
	private void checkIterations(int numOfIterations) {
		if(numOfIterations <= 0) {
			throw new IllegalArgumentException("Number of iterations [" + numOfIterations + "] can't be negative");
		}
	}
	
	// it would be good, if sizeOfMiniBatch would be multiple of 
	public void trainNetwork(int numOfIterations, int sizeOfMiniBatch) {
		checkIterations(numOfIterations);	
		if(sizeOfMiniBatch < database.getNumberOfClasses()) {
			LOGGER.warn("SizeOfiniBatch is too low and there is possibility to ... well IDK, it is just wrong");
			sizeOfMiniBatch = database.getNumberOfClasses();
		}
		for (int network = 0; network < networks.size(); network++) {
			long trainingStarted = System.currentTimeMillis();
			double[][] batch = new double[sizeOfMiniBatch][networks.get(network).getSizeOfInputVector()];
			double[][] targetValues = new double[sizeOfMiniBatch][];
			for (int i = 0; i < sizeOfMiniBatch; i++) {
				targetValues[i] = this.initializeTargetArray(network);
			}
			int[] learnedPatterns = new int[this.sizeOfOutputVector];
			int previousIndex = 0;
			for (int i = 0; i < numOfIterations ; i++) {
				for (int j = 0; j < sizeOfMiniBatch; j++) {
					LabelledItem item = database.getRandomizedLabelTrainingData();
					batch[j] = item.data;
					targetMaker.get(network).getTargetValues(item._class, previousIndex, targetValues[j]);
					previousIndex = item._class;
					learnedPatterns[previousIndex]++;
				}
				networks.get(network).miniBatchTraining(batch, targetValues);
			}
			long trainingEnded = System.currentTimeMillis();
			double sec = ((trainingEnded - trainingStarted) / 1000.0);
			LOGGER.info("Training: {} samples took {} seconds, {} samples/sec", (numOfIterations * sizeOfMiniBatch),
					sec, (numOfIterations * sizeOfMiniBatch)/sec);
			LOGGER.info("Learned Patterns: ");
			for (int i = 0; i < this.sizeOfOutputVector; i++) {
				LOGGER.info("{} - {}", i, learnedPatterns[i]);
			}
		}
	}
	
	public void trainNetwork(int numOfIterations) {
		checkIterations(numOfIterations);
		if(numOfIterations < database.getNumberOfClasses()) {
			LOGGER.warn("Count of iterations for training([{}]) is smaller than number of classes of division([{}]). "
					+ "Uneffective training possible: setting numOfIterations to numberOfClasses", numOfIterations, 
					database.getNumberOfClasses());
			numOfIterations = database.getNumberOfClasses();
		}
		for (int network = 0; network < networks.size(); network++) {
			long trainingStarted = System.currentTimeMillis();
			double[] targetValues = this.initializeTargetArray(network);
			int[] learnedPatterns = new int[this.sizeOfOutputVector];
			for (int i = 1; i < numOfIterations + 1; i++) {
				int newIndex = i % database.getNumberOfClasses();
				networks.get(network).feedForward(database.getTrainingIteratedData(newIndex).data);
				networks.get(network).backPropagation(targetMaker.get(network).getTargetValues(newIndex, 
						((i - 1) % database.getNumberOfClasses()), targetValues));
				learnedPatterns[newIndex]++;
			}
			long trainingEnded = System.currentTimeMillis();
			double sec = ((trainingEnded - trainingStarted) / 1000.0);
			LOGGER.info("Training: {} samples took {} seconds, {} samples/sec", numOfIterations, sec,
					numOfIterations/sec);
			LOGGER.info("Learned Patterns: ");
			for (int i = 0; i < this.sizeOfOutputVector; i++) {
				LOGGER.info("{} - {}", i, learnedPatterns[i]);
			}
		}
	}
}
