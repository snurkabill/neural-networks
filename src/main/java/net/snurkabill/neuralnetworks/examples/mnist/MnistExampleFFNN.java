package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.FFClassificationNetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.RBMClassificationNetworkManager;
import net.snurkabill.neuralnetworks.math.function.transferfunction.*;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToRectifiedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
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

    public static void basicBenchmark() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);

        TransferFunctionCalculator calculator1 = new ParametrizedHyperbolicTangens();
        NewDatabase database = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), calculator1, "MNIST", seed);
        List<Integer> topology = new ArrayList<>();
        topology.add(database.getVectorSize());
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(database.getTargetSize());

        double weightsScale = 1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                Arrays.asList(calculator1, calculator1),
                /*new GaussianRndWeightsFactory(weightsScale, seed)*/
                new SmartGaussianRndWeightsFactory(calculator1, seed),
                FeedForwardHeuristic.createDefaultHeuristic());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        TransferFunctionCalculator calculator2 = new SigmoidFunction();
        NewDatabase database2 = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), calculator2, "MNIST", seed);
        BoltzmannMachineHeuristic heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.l2RegularizationConstant = 0.99999;
        RestrictedBoltzmannMachine machine =
                new BinaryToBinaryRBM("RBM 1",
                        (database.getVectorSize() + database.getTargetSize()), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new RBMClassificationNetworkManager(machine, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        machine =
                new BinaryToRectifiedRBM("RBM 2",
                        (database.getVectorSize() + database.getTargetSize()), 500,
                        new GaussianRndWeightsFactory(0.1, seed),
                        heuristic, seed);
        NetworkManager manager_rbm2 = new RBMClassificationNetworkManager(machine, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.5;
        heuristic.temperature = 0.8;
        machine =
                new BinaryToBinaryRBM("RBM 3",
                        (database.getVectorSize() + database.getTargetSize() ), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm3 = new RBMClassificationNetworkManager(machine, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 10;
        heuristic.momentum = 0.5;
        heuristic.temperature = 0.8;
        machine = new BinaryToBinaryRBM("RBM 4",
                        (database.getVectorSize() + database.getTargetSize()), 200,
                        new GaussianRndWeightsFactory(0.01, seed),
                        heuristic, seed);
        NetworkManager manager_rbm4 = new RBMClassificationNetworkManager(machine, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(manager, manager_rbm/*, manager_rbm2, manager_rbm3, manager_rbm4*/), 2);
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(5, 10000, superManager, "basicMnistBenchmark");
        benchmarker.benchmark();
    }

    public static void startExample() throws IOException {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);

        List<Integer> topology = new ArrayList<>();
        topology.add(INPUT_SIZE);
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(OUTPUT_SIZE);

        TransferFunctionCalculator calculator1 = new ParametrizedHyperbolicTangens();
        NewDatabase database = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), calculator1, "MNIST", seed);

        double weightsScale = 1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                Arrays.asList(calculator1, calculator1),
                new SmartGaussianRndWeightsFactory(calculator1, seed),
                FeedForwardHeuristic.createDefaultHeuristic());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        LOGGER.info("Process started!");
        int sizeOfTrainingBatch = database.getTrainingSetSize();
        for (int i = 0; i < 10; i++) {
            manager.trainNetwork(sizeOfTrainingBatch);
            manager.validateNetwork();
            SupervisedNetworkResults results = (SupervisedNetworkResults) manager.getTestResults();
            LOGGER.info("Success rate is {} in {} iteration", results.getSuccessPercentage(), i);
        }

        manager.testNetwork();
        SupervisedNetworkResults results = (SupervisedNetworkResults) manager.getTestResults();
        LOGGER.info("Success rate is {}", results.getSuccessPercentage());

    }


    public static void linearVsTanhOutput() {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);

        List<Integer> topology = new ArrayList<>();
        topology.add(INPUT_SIZE);
        for (int i = 0; i < hiddenLayers.length; i++) {
            topology.add(hiddenLayers[i]);
        }
        topology.add(OUTPUT_SIZE);

        TransferFunctionCalculator tanh = new ParametrizedHyperbolicTangens();
        NewDatabase database = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), tanh, "MNIST", seed);

        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("tanh", topology,
                Arrays.asList(tanh, tanh),
                new SmartGaussianRndWeightsFactory(tanh, seed),
                FeedForwardHeuristic.createDefaultHeuristic());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        TransferFunctionCalculator linear = new LinearFunction();
        NewDatabase database2 = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), linear, "MNIST", seed);

        OnlineFeedForwardNetwork network2 = new OnlineFeedForwardNetwork("linear", topology,
                Arrays.asList(tanh, linear),
                new SmartGaussianRndWeightsFactory(linear, seed),
                FeedForwardHeuristic.createDefaultHeuristic());
        NetworkManager manager2 = new FFClassificationNetworkManager(network2, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        MasterNetworkManager master = new MasterNetworkManager("asdf", Arrays.asList(/*manager, */manager2), 1);

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

    public static void useRelu() {
        long seed = 0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        List<Integer> topology = new ArrayList<>();
        topology.add(INPUT_SIZE);
        topology.add(200);
        topology.add(50);
        topology.add(OUTPUT_SIZE);
        NewDatabase database = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), new ParametrizedHyperbolicTangens(), "MNIST", seed);

        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("trans", topology,
                Arrays.asList(new SoftPlus(), new SoftPlus(), new ParametrizedHyperbolicTangens()),
                new GaussianRndWeightsFactory(0.1, seed),
                FeedForwardHeuristic.littleStepsWithHugeMoment());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        OnlineFeedForwardNetwork network2 = new OnlineFeedForwardNetwork("trans2", topology,
                Arrays.asList(new ParametrizedHyperbolicTangens(), new ParametrizedHyperbolicTangens(), new ParametrizedHyperbolicTangens()),
                new GaussianRndWeightsFactory(0.1, seed),
                FeedForwardHeuristic.littleStepsWithHugeMoment());
        NetworkManager manager2 = new FFClassificationNetworkManager(network2, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        MasterNetworkManager master = new MasterNetworkManager("asdf", Arrays.asList(manager, manager2), 2);

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(100, 100, master, "softmaxVsHyperTan");
        benchmarker.benchmark();
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
