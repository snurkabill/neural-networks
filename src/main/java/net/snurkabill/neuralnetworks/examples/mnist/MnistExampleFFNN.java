package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.FFNN.FeedForwardNetworkBenchmarker;
import net.snurkabill.neuralnetworks.benchmark.FFNN.MiniBatchVsOnlineBenchmarker;
import net.snurkabill.neuralnetworks.data.DataItem;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.data.MnistDatasetReader;
import net.snurkabill.neuralnetworks.deepnets.StuckedRBM;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.energybasednetwork.HeuristicParamsRBM;
import net.snurkabill.neuralnetworks.energybasednetwork.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.feedforwardnetwork.manager.FeedForwardNetworkOfflineManager;
import net.snurkabill.neuralnetworks.feedforwardnetwork.core.FeedForwardNeuralNetwork;
import net.snurkabill.neuralnetworks.feedforwardnetwork.heuristic.HeuristicParamsFFNN;
import net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction.HyperbolicTangens;
import net.snurkabill.neuralnetworks.feedforwardnetwork.trasferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.PretrainedWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.WeightsFactory;
import net.snurkabill.neuralnetworks.results.SupervisedTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;

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
		
		MiniBatchVsOnlineBenchmarker benchmarker = new MiniBatchVsOnlineBenchmarker(manager, 10, 500, 50, miniManager);
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

        LOGGER.info("Deep learning!!!");
        long seed = 0;
        double weightsScale = 0.001;
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
        machines.add(new BinaryRestrictedBoltzmannMachine(database.getSizeOfVector(), 50,
                new GaussianRndWeightsFactory(weightsScale, seed), HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        machines.add(new BinaryRestrictedBoltzmannMachine(50, 20,
                new GaussianRndWeightsFactory(weightsScale, seed), HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        machines.add(new BinaryRestrictedBoltzmannMachine(20, 20,
                new GaussianRndWeightsFactory(weightsScale, seed), HeuristicParamsRBM.createBasicHeuristicParams(), seed));
        StuckedRBM rbms = new StuckedRBM(machines, "PrvniKominEver");
        int numOfIterations = 1000;
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
        List<Integer> topology = Arrays.asList(784, 50, 20, 20, 10);

        List<FeedForwardNeuralNetwork> networks = new ArrayList<>();

        HeuristicParamsFFNN heuristic = HeuristicParamsFFNN.createDefaultHeuristic();
        networks.add(new FeedForwardNeuralNetwork(topology, wFactory, "eta=0.001",
                heuristic, new ParametrizedHyperbolicTangens(), seed));
        heuristic = HeuristicParamsFFNN.createDefaultHeuristic();
        heuristic.eta = 0.01;
        networks.add(new FeedForwardNeuralNetwork(topology, wFactory, "experimental",
                heuristic, new ParametrizedHyperbolicTangens(), seed));
        heuristic = HeuristicParamsFFNN.createDefaultHeuristic();
        heuristic.eta = 0.01;
        networks.add(new FeedForwardNeuralNetwork(topology, wFactory, "eta=0.01",
                heuristic, new ParametrizedHyperbolicTangens(), seed));

        for (int i = 0; i < networks.size() /**- 1 /** -1 because last networks represents simple back prop**/; i++) {
            for (int j = 0; j < machines.size(); j++) {
                networks.get(i).setWeights(j, machines.get(j).createLayerForFFNN());
            }
        }
        FeedForwardNetworkOfflineManager manager = new FeedForwardNetworkOfflineManager(networks, database, false);
        FeedForwardNetworkBenchmarker benchmark = new FeedForwardNetworkBenchmarker(manager, 50, 2000);
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
