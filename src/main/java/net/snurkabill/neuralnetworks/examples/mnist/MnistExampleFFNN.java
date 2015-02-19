package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
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
    public static final String FILE_PATH = "src/main/resources/examples/mnist/data/";
    public static final int FULL_MNIST_SIZE = 60_000;
    public static final int INPUT_SIZE = 784;
    public static final int OUTPUT_SIZE = 10;
    public static final int[] hiddenLayers = {200};

    public static void hugeRBMBenchmark() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

        double weightsScale = 0.01;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh",
                Arrays.asList(database.getSizeOfVector(), 200, database.getNumberOfClasses()),
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        int validationIters = 5;
        BoltzmannMachineHeuristic heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.batchSize = 5;
        heuristic.constructiveDivergenceIndex = 3;
        machine = new BinaryRestrictedBoltzmannMachine("RBM heurCalculator",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm2 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        machine = new BinaryRestrictedBoltzmannMachine("RBM myHeuristic_low",
                (database.getSizeOfVector() + database.getNumberOfClasses()), 2000,
                new GaussianRndWeightsFactory(weightsScale, seed),
                heuristic, seed);
        NetworkManager manager_rbm3 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(validationIters,
                        database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.batchSize = 5;
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
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);
        List<Integer> topology = new ArrayList<>();
        topology.add(database.getSizeOfVector());
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(database.getNumberOfClasses());

        double weightsScale = 1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                /*new GaussianRndWeightsFactory(weightsScale, seed)*/
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        BoltzmannMachineHeuristic heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM 1",
                        (database.getSizeOfVector()), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new UnsupervisedRBMManager(machine, database, seed, null,
                new ProbabilisticAssociationVectorValidator(3));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        machine = new BinaryRestrictedBoltzmannMachine("RBM 3",
                (database.getSizeOfVector()), 200,
                new GaussianRndWeightsFactory(0.01, seed),
                heuristic, seed);
        NetworkManager manager_rbm2 = new UnsupervisedRBMManager(machine, database, seed, null,
                new ProbabilisticAssociationVectorValidator(3));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.1;
        heuristic.temperature = 0.2;
        machine = new BinaryRestrictedBoltzmannMachine("RBM 10",
                (database.getSizeOfVector()), 200,
                new GaussianRndWeightsFactory(0.01, seed),
                heuristic, seed);
        NetworkManager manager_rbm3 = new UnsupervisedRBMManager(machine, database, seed, null,
                new ProbabilisticAssociationVectorValidator(3));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.1;
        heuristic.temperature = 0.05;
        machine = new BinaryRestrictedBoltzmannMachine("RBM 50",
                (database.getSizeOfVector()), 200,
                new GaussianRndWeightsFactory(0.01, seed),
                heuristic, seed);
        NetworkManager manager_rbm4 = new UnsupervisedRBMManager(machine, database, seed, null,
                new ProbabilisticAssociationVectorValidator(3));

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(/*manager,*/ manager_rbm, manager_rbm2, manager_rbm3, manager_rbm4));
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
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());

        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

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
