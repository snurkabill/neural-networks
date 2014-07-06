package net.snurkabill.neuralnetworks.examples.binarymarketmove;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import static net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager.LOGGER;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.HyperbolicTangens;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;

public class BinaryMarketMove {
	
	public static void startExample() throws IOException {
		long seed = 0;
		
		Database database = Database.loadDatabase(new File("TwoSecBinaryMarketMove"));
		int inputSize = database.getSizeOfVector();
		int outputSize = database.getNumberOfClasses();
		int[] hiddenLayers = {20, 5};
	
		List<Integer> topology = new ArrayList<>();
		topology.add(inputSize);
		for (int i = 0; i < hiddenLayers.length; i++) {
			topology.add(hiddenLayers[i]);
		}
		topology.add(outputSize);
		
		double weightsScale = 0.001;
		FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "BinaryMove", 
				HeuristicParamsFFNN.deepProgressive(), new HyperbolicTangens(), seed);
		
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(
				Collections.singletonList(network), database);
		
		int sizeOfPretrainingBatch = 20;
		manager.pretrainInputNeurons(sizeOfPretrainingBatch);
		LOGGER.info("Process started!");
		int sizeOfTrainingBatch = 100_000;
		for (int i = 0; i < 100; i++) {
			manager.trainNetwork(sizeOfTrainingBatch);
			SupervisedTestResults results = (SupervisedTestResults) manager.testNetworkOnWholeTestingDataset().get(0);
			LOGGER.info("Success rate is {} in {} iteration", results.getSuccessPercentage(), i);
		}
    }
	
}
