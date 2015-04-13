package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.BasicHeuristic;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToRectifiedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
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
        Database database = createDatabase(2000, false, seed);
        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 200, database.getNumberOfClasses());
        BasicHeuristic heuristic = new FeedForwardHeuristic().setLearningRate(0.01).setMomentum(0.1);
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                (FeedForwardHeuristic) heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals(71.0, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    @Test
    public void learningBinaryBinaryRBM() {
        long seed = 0;
        double weightsScale = 0.01;
        Database database = createDatabase(2000, true, seed);
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 3;
        BinaryToBinaryRBM machine =
                new BinaryToBinaryRBM("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals(62.0, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    @Test
    public void learningBinaryRectifiedRBM() {
        long seed = 0;
        double weightsScale = 0.1;
        Database database = createDatabase(2000, true, seed);
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 10;
        BinaryToBinaryRBM machine =
                new BinaryToRectifiedRBM("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals(51.5, manager.getTestResults().getComparableSuccess(), 0.0001);
    }

    private Database createDatabase(int numOfElements, boolean binary, long seed) {
        File labels = new File("src/test/resources/mnist/train-labels-idx1-ubyte.gz");
        File images = new File("src/test/resources/mnist/train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images, numOfElements, binary);
            return new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
    }
}
