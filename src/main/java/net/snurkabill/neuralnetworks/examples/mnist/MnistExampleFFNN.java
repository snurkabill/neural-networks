package net.snurkabill.neuralnetworks.examples.mnist;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.data.MnistDatasetReader;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.feedforwardnetwork.HyperbolicTangens;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistExampleFFNN {

	public static final Logger LOGGER = LoggerFactory.getLogger("MnistExample FFNN");
	
	public static final int INPUT_SIZE = 784;
	public static final int OUTPUT_SIZE = 10;
	public static final int[] hiddenLayers = {200};
	
	public static void startExample() throws IOException {
		long seed = 0;
		File labels = new File("target/minst/train-labels-idx1-ubyte.gz");
		File images = new File("target/minst/train-images-idx3-ubyte.gz");
		MnistDatasetReader reader;
		try {
			reader = new MnistDatasetReader(labels, images);
		} catch (IOException ex) {
			throw new IllegalArgumentException(ex);
		}
		
		List<Integer> topology = new ArrayList<>();
		topology.add(INPUT_SIZE);
		for (int i = 0; i < hiddenLayers.length; i++) {
			topology.add(hiddenLayers[i]);
		}
		topology.add(OUTPUT_SIZE);
		
		double weightsScale = 0.001;
		FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "Clara:)", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database);
		
		int sizeOfPretrainingBatch = 100;
		manager.pretrainInputNeurons(sizeOfPretrainingBatch);
		LOGGER.info("Process started!");
		int sizeOfTrainingBatch = 10_000;
		for (int i = 0; i < 100; i++) {
			manager.trainNetwork(sizeOfTrainingBatch);
			SupervisedTestResults results = (SupervisedTestResults) manager.testNetworkOnWholeTestingDataset().get(0);
			LOGGER.info("Success rate is {} in {} iteration", results.getSuccessPercentage(), i);
		}
    }	
}
