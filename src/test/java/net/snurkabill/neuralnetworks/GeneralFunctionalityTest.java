package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.FFClassificationNetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.RBMClassificationNetworkManager;
import net.snurkabill.neuralnetworks.math.function.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToRectifiedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.newdata.database.RegressionDatabase;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class GeneralFunctionalityTest {

    @Test
    public void learningFeedForward() {
        long seed = 0;
        TransferFunctionCalculator calculator = new ParametrizedHyperbolicTangens();
        NewDatabase database = createClassificationDatabase(2000, false, seed, calculator);
        List<Integer> topology = Arrays.asList(database.getVectorSize(), 200, database.getTargetSize());
        BasicHeuristic heuristic = new FeedForwardHeuristic().setLearningRate(0.01).setMomentum(0.1);
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                (FeedForwardHeuristic) heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(1000);
        manager.testNetwork();
        assertEquals(71.0, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    @Test
    public void learningBinaryBinaryRBM() {
        long seed = 0;
        double weightsScale = 0.01;
        NewDatabase database = createClassificationDatabase(2000, true, seed, new SigmoidFunction());
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 3;
        BinaryToBinaryRBM machine =
                new BinaryToBinaryRBM("RBM basicHeuristic",
                        (database.getVectorSize() + database.getTargetSize()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new RBMClassificationNetworkManager(machine, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(1000);
        manager.testNetwork();
        assertEquals(62.0, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    @Test
    public void learningBinaryRectifiedRBM() {
        long seed = 0;
        double weightsScale = 0.1;
        NewDatabase database = createClassificationDatabase(2000, true, seed, new SigmoidFunction());
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.01;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 1;
        BinaryToBinaryRBM machine =
                new BinaryToRectifiedRBM("RBM basicHeuristic",
                        (database.getVectorSize() + database.getTargetSize()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new RBMClassificationNetworkManager(machine, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(1000);
        manager.testNetwork();
        assertEquals(74.5, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    private NewDatabase createClassificationDatabase(int numOfElements, boolean binary, long seed,
                                                     TransferFunctionCalculator calculator) {
        File labels = new File("src/test/resources/mnist/train-labels-idx1-ubyte.gz");
        File images = new File("src/test/resources/mnist/train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images, numOfElements, binary);
            return new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(), reader.getTestingData(),
                    calculator, "MNIST", seed);
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
    }

    private NewDatabase createRegressionDatabase(int numOfElements, boolean binary, long seed,
                                                     TransferFunctionCalculator calculator) {
        File labels = new File("src/test/resources/mnist/train-labels-idx1-ubyte.gz");
        File images = new File("src/test/resources/mnist/train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images, numOfElements, binary);
            return new RegressionDatabase(reader.getRegressionTrainingData(), reader.getRegressionValidationData(),
                    reader.getRegressionTestingData(), "MNIST", seed);
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
    }
}
