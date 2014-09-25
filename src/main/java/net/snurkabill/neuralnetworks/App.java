package net.snurkabill.neuralnetworks;

import java.io.IOException;

import net.snurkabill.neuralnetworks.examples.mnist.MnistExampleFFNN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {

	private static final Logger LOGGER = LoggerFactory.getLogger("main");

	public static void main(String argv[]) throws IOException {

		//BinaryMarketMove.startExample();
		
		//MnistExampleFFNN.startExample();
        MnistExampleFFNN.deepLearning();
		//MnistExampleFFNN.startSaveLoadExample();
		//MnistExampleFFNN.benchmarkOnMNIST();
		//new RBMTest().test();
		
		// *********************************************************************************
		// DBN test
		// *********************************************************************************
		/*double weightScale = 1;
		File labels = new File("minst/train-labels-idx1-ubyte.gz");
		File images = new File("minst/train-images-idx3-ubyte.gz");
		MnistDatasetReader reader;
		try {
			reader = new MnistDatasetReader(labels, images);
		} catch (IOException ex) {
			throw new IllegalArgumentException(ex);
		}
		
		Database database = new Database(0L, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		
		List<RestrictedBoltzmannMachine> machines = new ArrayList<>();
		
		machines.add(new BinaryRestrictedBoltzmannMachine(784, 300, new GaussianRndWeightsFactory(weightScale, 0),
				HeuristicParamsRBM.createBasicHeuristicParams(), 0));
		machines.add(new BinaryRestrictedBoltzmannMachine(300, 300, new GaussianRndWeightsFactory(weightScale, 0),
				HeuristicParamsRBM.createBasicHeuristicParams(), 0));
		machines.add(new BinaryRestrictedBoltzmannMachine(310, 1000, new GaussianRndWeightsFactory(weightScale, 0),
				HeuristicParamsRBM.createBasicHeuristicParams(), 0));
		
		DeepBeliefNeuralNetwork fuckingMonster = new DeepBeliefNeuralNetwork(machines, "Trhac Asfaltu");
		
		int numOfIterations = 5000;
		
		LOGGER.info("Inner training started");
		
		for (int i = 0; i < machines.size() - 1; i++) {
			for (int j = 0; j < numOfIterations; j++) {
				double[] input = database.getRandomizedTrainingData().data;
				LOGGER.trace("Size of input: {}", input.length);
				for (int k = 0; k < input.length; k++) {
					input[k] = input[k] > 30 ? 1 : 0;
				}
				fuckingMonster.trainInnerNetworkLevel(input, i);
                LOGGER.info("Iteration: {},  {}", i, j);
			}
		}
		
		LOGGER.info("Last layer training started");
		Random random = new Random(0);
		for (int i = 0; i < numOfIterations; i++) {
			int labelIndex = random.nextInt(10);
			double[] input = database.getTrainingData(labelIndex).data;
			double[] label = new double[10];
			label[labelIndex] = 1.0;
			for (int k = 0; k < input.length; k++) {
				input[k] = input[k] > 30 ? 1 : 0;
			}
			fuckingMonster.trainLastNetworkLayer(input, label);
		}
		
		LOGGER.info("Testing started");
		long testingStarted = System.currentTimeMillis();
		double[] targetValues = new double[10];
		
		int success = 0;
		int fail = 0;
		for (int i = 0; i < database.getNumberOfClasses(); i++) {
	
			for (int j = 0; j < targetValues.length; j++) {
				targetValues[j] = 0.0;
			}
			targetValues[i] = 1.0;
			
			for (int j = 0; j < database.getSizeOfTestingDataset(i); j++) {
				LOGGER.debug("testing {} iterations {}", j, database.getSizeOfTestingDataset(i));
				
				double[] input = database.getTestingIteratedData(i).data;
				for (int k = 0; k < input.length; k++) {
					input[k] = input[k] > 30 ? 1 : 0;
				}
				double[] output = fuckingMonster.feedForward(input, 3);
				
				//TODO : general evaluation 
				
				double max = -10000; 
				int bestIndex = -1;
				for (int omg = 0; omg < output.length; omg++) {
					if(output[omg] > max) { 
						max = output[omg]; 
						bestIndex = omg;
					}
				}
				if(bestIndex == i) {
					success++;
				} else fail++;
			}
		}
		long testingFinished = System.currentTimeMillis();
		int all = (success + fail);
		double sec = ((testingFinished - testingStarted) / 1000.0);
		LOGGER.info("Testing {} samples took {} seconds, {} samples/sec", all, sec, all/sec);
		LOGGER.info("RESULTS: {}", ((success * 100.0) / all));
        */

		// *********************************************************************************
		// benchmarker test
		// *********************************************************************************
		//MnistExampleFFNN.startExample();
		/*int seed = 0;
		double weightsScale = 0.0001;
		File labels = new File("target/minst/train-labels-idx1-ubyte.gz");
		File images = new File("target/minst/train-images-idx3-ubyte.gz");
		MnistDatasetReader reader;
		try {
			reader = new MnistDatasetReader(labels, images);
		} catch (IOException ex) {
			throw new IllegalArgumentException(ex);
		}

		List<FeedForwardNeuralNetwork> networks = new ArrayList<>();

		List<Integer> topology = new ArrayList<>();
		topology.add(784);
		topology.add(200);
		topology.add(10);

		networks.add(new FeedForwardNeuralNetwork(topology, new GaussianRndWeightsFactory(weightsScale, seed),
		 String.valueOf(0), HeuristicParamsFFNN.createDefaultHeuristic(), new HyperbolicTangens(), seed));
		
		 networks.add(new FeedForwardNeuralNetwork(topology, new GaussianRndWeightsFactory(weightsScale, seed),
		 String.valueOf(1), HeuristicParamsFFNN.createDefaultHeuristic(), new ParametrizedHyperbolicTangens(), seed));
		 
		Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
		FeedForwardNetworkOfflineManager manager
				= new FeedForwardNetworkOfflineManager(networks, database);

		FeedForwardNetworkBenchmarker benchmarker = new FeedForwardNetworkBenchmarker(manager, 100, 1000);

		benchmarker.benchmark();		*/
	}
}
