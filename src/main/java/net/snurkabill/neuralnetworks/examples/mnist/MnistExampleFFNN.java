package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.results.SupervisedNetworkResults;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

public class MnistExampleFFNN {

    public static final Logger LOGGER = LoggerFactory.getLogger("MnistExample FFNN");
    public static final String FILE_PATH = "../src/main/resources/mnist/";
    public static final int FULL_MNIST_SIZE = 60_000;
    public static final int INPUT_SIZE = 784;
    public static final int OUTPUT_SIZE = 10;
    public static final int[] hiddenLayers = {200};


    public static void hugeRBMBenchmark() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");

        double weightsScale = 0.01;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh",
                Arrays.asList(database.getSizeOfVector(), 200, database.getNumberOfClasses()),
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        int validationIters = 5;
        HeuristicRBM heuristic = HeuristicRBM.createStartingHeuristicParams();
        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = HeuristicRBM.createStartingHeuristicParams();
        heuristic.numOfTrainingIterations = 5;
        heuristic.constructiveDivergenceIndex = 3;
        machine = new BinaryRestrictedBoltzmannMachine("RBM heurCalculator",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm2 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = HeuristicRBM.createStartingHeuristicParams();
        machine = new BinaryRestrictedBoltzmannMachine("RBM myHeuristic_low",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 2000,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm3 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = HeuristicRBM.createStartingHeuristicParams();
        heuristic.numOfTrainingIterations = 5;
        heuristic.constructiveDivergenceIndex = 3;
        machine = new BinaryRestrictedBoltzmannMachine("RBM myHeuristic",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 2000,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm4 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(manager, manager_rbm, manager_rbm2, manager_rbm3, manager_rbm4));
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(60, 1000, superManager);
        benchmarker.benchmark();
    }

    public static void basicBenchmark() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        /*LOGGER.info("Building PCA");
        PCA pca = new PCA(Utilities.buildMatrix(reader.getTrainingData(), 10_000));
        LOGGER.info("Transforming Matrixces");
        Matrix trainingMatrix =
                pca.transform(Utilities.buildMatrix(reader.getTrainingData()), TransformationType.WHITENING);
        Matrix testingMatrix =
                pca.transform(Utilities.buildMatrix(reader.getTestingData()), TransformationType.WHITENING);
        LOGGER.info("Testing matrix sizes: {} {}", trainingMatrix.getRowDimension(), trainingMatrix.getColumnDimension());
        LOGGER.info("Testing matrix sizes: {} {}", testingMatrix.getRowDimension(), testingMatrix.getColumnDimension());
		*/
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");
        List<Integer> topology = new ArrayList<>();
        topology.add(database.getSizeOfVector());
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(database.getNumberOfClasses());

        double weightsScale = 0.01;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                /*new GaussianRndWeightsFactory(weightsScale, seed)*/
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        HeuristicRBM.createStartingHeuristicParams(), seed);
        NetworkManager manager_rbm = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(10, database.getNumberOfClasses()));

        /*HeuristicRBM heuristic = HeuristicRBM.createStartingHeuristicParams();
        machine = new BinaryRestrictedBoltzmannMachine("RBM heurCalculator",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 100,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm2 = new SupervisedRBMManager(machine, database, seed,null,
                new PartialProbabilisticAssociationVectorValidator(10, database.getNumberOfClasses()));
*/
        /*machine = new BinaryRestrictedBoltzmannMachine("RBM myHeuristic_low",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 150,
                        new GaussianRndWeightsFactory(weightsScale, seed), 
                        HeuristicRBM.createBasicHeuristicParams(), seed);
        NetworkManager manager_rbm3 = new BinaryRBMManager(machine, database, seed, 5, null);
		
		machine = new BinaryRestrictedBoltzmannMachine("RBM myHeuristic", 
				(database.getSizeOfVector() + database.getNumberOfClasses()), 200,
                        new GaussianRndWeightsFactory(weightsScale, seed), 
                        HeuristicRBM.createBasicHeuristicParams(), seed);
        NetworkManager manager_rbm4 = new BinaryRBMManager(machine, database, seed, 5, null);*/

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(manager, manager_rbm/*, manager_rbm2, manager_rbm3, manager_rbm4*/));
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(2, 1000, superManager);
        benchmarker.benchmark();
    }

    public static void startExample() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, false);

        List<Integer> topology = new ArrayList<>();
        topology.add(INPUT_SIZE);
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(OUTPUT_SIZE);

        double weightsScale = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("ParametrizedTanh", topology,
                new GaussianRndWeightsFactory(weightsScale, seed),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());

        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");

        FeedForwardNetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        LOGGER.info("Process started!");
        int sizeOfTrainingBatch = 1000;
        for (int i = 0; i < 10; i++) {
            manager.supervisedTraining(sizeOfTrainingBatch);
            manager.testNetwork();
            SupervisedNetworkResults results = (SupervisedNetworkResults) manager.getTestResults();
            LOGGER.info("Success rate is {} in {} iteration", results.getSuccessPercentage(), i);
        }
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

    public static MnistDatasetReader getReader(int numOfElements, boolean binary) {
        LOGGER.info("Unzipping files ...");
        File labels = new File(FILE_PATH + "train-labels-idx1-ubyte.gz");
        File images = new File(FILE_PATH + "train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images, numOfElements, binary);
            return reader;
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
    }
}
