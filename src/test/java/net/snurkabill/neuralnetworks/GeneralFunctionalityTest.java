package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.deep.BinaryDeepBeliefNetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.StuckedRBMTrainer;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.BinaryDeepBeliefNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.junit.Ignore;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.*;

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
                true, 61.5 == manager.getTestResults().getComparableSuccess());
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
        heuristic.batchSize = 3;
        BinaryRestrictedBoltzmannMachine machine =
                new BinaryRestrictedBoltzmannMachine("RBM basicHeuristic",
                        (database.getSizeOfVector() + database.getNumberOfClasses()), 50,
                        new GaussianRndWeightsFactory(weightsScale, seed),
                        heuristic, seed);
        NetworkManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(10, database.getNumberOfClasses()));
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                true, 63.0 == manager.getTestResults().getComparableSuccess());
    }

    private List<DataItem> createNoise() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

    private List<DataItem> createAnd() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        return list;
    }

    @Ignore
    @Test
    public void proofOfConcept() {
        long seed = 0;
        double weightsScale = 0.01;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();

        trainingSet.put(0, createAnd());
        trainingSet.put(1, createNoise());

        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            and.addAll(createAnd());
            noise.addAll(createNoise());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);

        Database database = new Database(0, trainingSet, testingSet, "Database");

        HeuristicDBN heuristic = new HeuristicDBN();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 5;
        heuristic.numofRunsBaseRBMItself = 1;
        heuristic.numOfRunsOfNetworkChimney = 1;
        List<RestrictedBoltzmannMachine> machines = new ArrayList<>(2);
        machines.add(new BinaryRestrictedBoltzmannMachine("InnerRBM", database.getSizeOfVector(), 2,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        machines.add(new BinaryRestrictedBoltzmannMachine("InnerRBM2", 2, 2,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        StuckedRBM stuckedRBMs = new StuckedRBM(machines, "Stucked RBMs");
        StuckedRBMTrainer.train(stuckedRBMs, database, 100, 100);
        BinaryDeepBeliefNetwork DBN = new BinaryDeepBeliefNetwork("DBN", database.getNumberOfClasses(), 1,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, 0, stuckedRBMs);
        NetworkManager manager = new BinaryDeepBeliefNetworkManager(DBN, database, null);
        manager.supervisedTraining(100);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                0.0, manager.getTestResults().getComparableSuccess(), 0.00001);
    }

    @Test
    public void learningBinaryDBN() {
        long seed = 0;
        double weightsScale = 0.01;
        Database database = createDatabase(60_000, true);
        HeuristicDBN heuristic = new HeuristicDBN();
        heuristic.momentum = 0.1;
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 5;
        List<RestrictedBoltzmannMachine> machines = new ArrayList<>(2);
        machines.add(new BinaryRestrictedBoltzmannMachine("InnerRBM", database.getSizeOfVector(), 500,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        machines.add(new BinaryRestrictedBoltzmannMachine("InnerRBM2", 500, 500,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));

        StuckedRBM stuckedRBMs = new StuckedRBM(machines, "Stucked RBM");
        StuckedRBMTrainer.train(stuckedRBMs, database, 100000, 100000);

        BinaryDeepBeliefNetwork DBN = new BinaryDeepBeliefNetwork("DBN", database.getNumberOfClasses(), 2000,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, 0, stuckedRBMs);

        NetworkManager manager = new BinaryDeepBeliefNetworkManager(DBN, database, null);
        manager.supervisedTraining(10000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                true, 63.0 == manager.getTestResults().getComparableSuccess());
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
