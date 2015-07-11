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
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToRectifiedRBM;
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
    public static final int[] hiddenLayers = {200};

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
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.l2RegularizationConstant = 0.99999;
        RestrictedBoltzmannMachine machine =
                new BinaryToBinaryRBM("RBM 1",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        machine =
                new BinaryToRectifiedRBM("RBM 2",
                        (database.getSizeOfVector() + database.getNumberOfClasses() ), 500,
                        new GaussianRndWeightsFactory(0.1, seed),
                        heuristic, seed);
        NetworkManager manager_rbm2 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.5;
        heuristic.temperature = 0.8;
        machine =
                new BinaryToBinaryRBM("RBM 3",
                        (database.getSizeOfVector() + database.getNumberOfClasses() ), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm3 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.5;
        heuristic.temperature = 0.8;
        machine = new BinaryToBinaryRBM("RBM 4",
                        (database.getSizeOfVector() + database.getNumberOfClasses() ), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm4 = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(manager, manager_rbm/*, manager_rbm2, manager_rbm3, manager_rbm4*/), 2);
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(5, 10000, superManager);
        benchmarker.benchmark();
    }

    public static void startExample() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, false);

        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);
                /*.stochasticStandardNormalization(1.0);*/

        List<Integer> topology = new ArrayList<>();
        topology.add(database.getSizeOfVector());
        /*for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }*/
        topology.add(database.getNumberOfClasses());

        double weightsScale = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("ParametrizedTanh", topology,
                new GaussianRndWeightsFactory(weightsScale, seed),
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());

        FeedForwardNetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        LOGGER.info("Process started!");
        int sizeOfTrainingBatch = database.getTrainingSetSize();
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
