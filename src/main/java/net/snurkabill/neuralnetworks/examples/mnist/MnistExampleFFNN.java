package net.snurkabill.neuralnetworks.examples.mnist;

import java.io.File;
import java.io.IOException;
import java.util.*;

import net.snurkabill.neuralnetworks.benchmark.FFNN.FeedForwardNetworkBenchmarker;
import net.snurkabill.neuralnetworks.benchmark.FFNN.MiniBatchVsOnlineBenchmarker;
import net.snurkabill.neuralnetworks.data.DataItem;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.data.MnistDatasetReader;
import net.snurkabill.neuralnetworks.deepnets.StuckedRBM;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.energybasednetwork.HeuristicParamsRBM;
import net.snurkabill.neuralnetworks.energybasednetwork.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.feedforwardnetwork.HyperbolicTangens;
import net.snurkabill.neuralnetworks.feedforwardnetwork.SigmoidFunction;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.PretrainedWeightsFactory;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.WeightsFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MnistExampleFFNN {

	public static final Logger LOGGER = LoggerFactory.getLogger("MnistExample FFNN");
	public static final String FILE_PATH = "minst/";
	public static final int INPUT_SIZE = 784;
	public static final int OUTPUT_SIZE = 10;
	public static final int[] hiddenLayers = {200};
	
	public static void benchmarkOnMNIST() throws IOException {
		long seed = 0;
		MnistDatasetReader reader = getReader();

		List<Integer> topology = new ArrayList<>();
		topology.add(INPUT_SIZE);
		for (int i = 0; i < hiddenLayers.length; i++) {
			topology.add(hiddenLayers[i]);
		}
		topology.add(OUTPUT_SIZE);
		
		HeuristicParamsFFNN heuristic = HeuristicParamsFFNN.createDefaultHeuristic();

		double weightsScale = 0.001;
		FeedForwardNeuralNetwork network = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "Online_learning_simple", 
				HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		FeedForwardNeuralNetwork network2 = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "Online_learning", 
				heuristic, new HyperbolicTangens(), seed);
		
		FeedForwardNeuralNetwork networkMiniBatch = new FeedForwardNeuralNetwork(topology, 
				new GaussianRndWeightsFactory(weightsScale, seed), "miniBatch_learning", 
				heuristic, new HyperbolicTangens(), seed);
		
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Arrays.asList(network, network2), database, false);
		FeedForwardNetworkOfflineManager miniManager = new FeedForwardNetworkOfflineManager(Collections.singletonList(networkMiniBatch), database, false);
		
		MiniBatchVsOnlineBenchmarker benchmarker = new MiniBatchVsOnlineBenchmarker(manager, 10, 500, 50, miniManager, 20);
		benchmarker.benchmark();
	}
	
	public static void startExample() throws IOException {
		long seed = 0;
        MnistDatasetReader reader = getReader();
		
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
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database, false);
		
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

    public static void convertMap(Map<Integer, List<DataItem>> map) {
        LOGGER.info("Converting map");
        for (Map.Entry<Integer, List<DataItem>> entry : map.entrySet()) {
            for (DataItem item : entry.getValue()) {
                for (int i = 0; i < item.data.length; i++) {
                    item.data[i] = item.data[i] < 30 ? 0.0 : 1.0;
                }
            }
        }
    }

    public static void deepLearning() throws IOException {
        long seed = 0;
        double weightsScale = 1;
        MnistDatasetReader reader = getReader();

        Map<Integer, List<DataItem>> testingData = reader.getTestingData();
        Map<Integer, List<DataItem>> trainingData = reader.getTrainingData();

        convertMap(testingData);
        convertMap(trainingData);

        Database database = new Database(seed, trainingData, testingData, "database");

        List<RestrictedBoltzmannMachine> machines = new ArrayList<>();
        /*machines.add(new BinaryRestrictedBoltzmannMachine(database.getSizeOfVector(), 200,
                new GaussianRndWeightsFactory(weightsScale, seed),
                HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        machines.add(new BinaryRestrictedBoltzmannMachine(200, database.getNumberOfClasses(),
                new GaussianRndWeightsFactory(weightsScale, seed), 
                HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        */
        machines.add(new BinaryRestrictedBoltzmannMachine(database.getSizeOfVector(), database.getNumberOfClasses(),
                new GaussianRndWeightsFactory(weightsScale, seed), HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        StuckedRBM rbms = new StuckedRBM(machines, "PrvniKominEver");

        int numOfIterations = 10;
        Iterator<DataItem> iterator = database.getInfiniteTrainingIterator();
        for (int i = 0; i < rbms.getNumOfLevels(); i++) {
            for (int j = 0; j < numOfIterations; j++) {
                rbms.trainNetworkLevel(iterator.next().data, i);
            }
        }
        List<double[][]> weights = new ArrayList<>();
        for (int i = 0; i < machines.size(); i++) {
            weights.add(machines.get(i).createLayerForFFNN());
        }
        WeightsFactory pretrainedFactory = new PretrainedWeightsFactory(weights);
        WeightsFactory wFactory = new GaussianRndWeightsFactory(weightsScale, seed);
        List<Integer> topology = Arrays.asList(784, 10);
        FeedForwardNeuralNetwork myNetwork = new FeedForwardNeuralNetwork(topology, pretrainedFactory,
                "AlreadyPretrained", HeuristicParamsFFNN.createDefaultHeuristic(), new SigmoidFunction(), seed);
        FeedForwardNeuralNetwork myNetwork2 = new FeedForwardNeuralNetwork(topology, wFactory, "nonPretrained",
                HeuristicParamsFFNN.createDefaultHeuristic(), new SigmoidFunction(), seed);

        FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Arrays.asList(myNetwork, myNetwork2),
                database, false);

        FeedForwardNetworkBenchmarker benchmark = new FeedForwardNetworkBenchmarker(manager, 100, 5000, 0);
        benchmark.benchmark();
    }

    public static MnistDatasetReader getReader() {
        File labels = new File(FILE_PATH + "train-labels-idx1-ubyte.gz");
        File images = new File(FILE_PATH + "train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images);
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
        return reader;
    }
	
	public static void startSaveLoadExample() throws IOException {
		long seed = 0;

        MnistDatasetReader reader = getReader();
		FeedForwardNeuralNetwork network = FeedForwardNeuralNetwork.buildNeuralNetwork(new File("saveTry"), "Clara:)",
                HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed);
		
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(Collections.singletonList(network), database, false);
		
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
