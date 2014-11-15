package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.binary.SupervisedBinaryRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.BinaryRestrictedBoltzmannMachine;
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
        Database database = createDatabase(2000, false);
        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 200, database.getNumberOfClasses());
        FFNNHeuristic heuristic = new FFNNHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                true, 60 < manager.getTestResults().getComparableSuccess());
    }

    @Test
    public void learningBinaryRBM() {
        long seed = 0;
        double weightsScale = 0.01;
        Database database = createDatabase(2000, true);
        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.numOfTrainingIterations = 3;
        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new SupervisedBinaryRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(10, database.getNumberOfClasses()));
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                true, 60 < manager.getTestResults().getComparableSuccess());
    }

    private Database createDatabase(int numOfElements, boolean binary) {
        File labels = new File("src/test/resources/mnist/train-labels-idx1-ubyte.gz");
        File images = new File("src/test/resources/mnist/train-images-idx3-ubyte.gz");
        MnistDatasetReader reader;
        try {
            reader = new MnistDatasetReader(labels, images, numOfElements, binary);
            return new Database(0, reader.getTrainingData(), reader.getTestingData(), "MNIST");
        } catch (IOException ex) {
            throw new IllegalArgumentException(ex);
        }
    }
}
