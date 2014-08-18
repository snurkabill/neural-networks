package net.snurkabill.neuralnetworks.examples.mnist;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.snurkabill.neuralnetworks.benchmark.MiniBatchVsOnlineBenchmarker;
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
	
	public static void benchmarkOnMNIST() throws IOException {
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
				new GaussianRndWeightsFactory(weightsScale, seed), "Online_learning", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		FeedForwardNeuralNetwork networkMiniBatch = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "miniBatch_learning", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database);
		FeedForwardNetworkOfflineManager miniManager = new FeedForwardNetworkOfflineManager(Collections.singletonList(networkMiniBatch), database);
		
		
		MiniBatchVsOnlineBenchmarker benchmarker = new MiniBatchVsOnlineBenchmarker(manager, 10, 500, 50, miniManager);
		benchmarker.benchmark();
	}
	
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
		
		int sizeOfPretrainingBatch = 10;
		manager.pretrainInputNeurons(sizeOfPretrainingBatch);
		LOGGER.info("Process started!");
		int sizeOfTrainingBatch = 10_000;
		for (int i = 0; i < 10; i++) {
			manager.trainNetwork(sizeOfTrainingBatch);
			SupervisedTestResults results = (SupervisedTestResults) manager.testNetworkOnWholeTestingDataset().get(0);
			LOGGER.info("Success rate is {} in {} iteration", results.getSuccessPercentage(), i);
		}
		
		/*manager.getNetworkList().get(0).saveNetwork(new File("saveTry"));
		
		FeedForwardNeuralNetwork network2 = FeedForwardNeuralNetwork.buildNeuralNetwork(new File("saveTry"), "mana", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		FeedForwardNetworkOfflineManager manager2 = new FeedForwardNetworkOfflineManager(
				Collections.singletonList(network2), database);
		
		SupervisedTestResults results = (SupervisedTestResults) manager2.testNetworkOnWholeTestingDataset().get(0);
		LOGGER.info("Success rate is {}", results.getSuccessPercentage());*/
    }	
	
	public static void startSaveLoadExample() throws IOException {
		long seed = 0;
		File labels = new File("target/minst/train-labels-idx1-ubyte.gz");
		File images = new File("target/minst/train-images-idx3-ubyte.gz");
		MnistDatasetReader reader;
		try {
			reader = new MnistDatasetReader(labels, images);
		} catch (IOException ex) {
			throw new IllegalArgumentException(ex);
		}

		FeedForwardNeuralNetwork network = FeedForwardNeuralNetwork.buildNeuralNetwork(new File("saveTry"), "Clara:)", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database);
		
		LOGGER.info("Process started!");
		
		SupervisedTestResults results = (SupervisedTestResults) manager.testNetworkOnWholeTestingDataset().get(0);
		LOGGER.info("Success rate is {}", results.getSuccessPercentage());
		
		/*
		List<Integer> topology = new ArrayList<>();
		topology.add(2);
		topology.add(2);
		topology.add(1);
		double weightsScale = 1;
		long seed = 1;
		
		FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "Clara:)", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		network.saveNetwork(new File("asdf"));
		
		FeedForwardNeuralNetwork network2 = FeedForwardNeuralNetwork.buildNeuralNetwork(new File("asdf"), "Clara:)", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		network2.getName();
	*/	
	}
}
