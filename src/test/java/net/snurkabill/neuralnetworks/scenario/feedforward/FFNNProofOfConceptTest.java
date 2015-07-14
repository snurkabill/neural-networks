package net.snurkabill.neuralnetworks.scenario.feedforward;

import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.FFClassificationNetworkManager;
import net.snurkabill.neuralnetworks.math.function.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.scenario.LogicFunctions;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.junit.Test;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class FFNNProofOfConceptTest extends LogicFunctions {

    @Test
    public void and() {
        long seed = 0;
        NewDataItem[][] trainingSet = new NewDataItem[2][3];
        trainingSet[0] = createAnd();
        trainingSet[1] = createNotAnd();

        NewDataItem[][] validationSet = new NewDataItem[2][3];
        validationSet[0] = createAnd();
        validationSet[1] = createNotAnd();

        NewDataItem[][] testingSet = new NewDataItem[2][3];
        testingSet[0] = createAnd();
        testingSet[1] = createNotAnd();
        NewDatabase database = new ClassificationDatabase(trainingSet, validationSet, testingSet, new SigmoidFunction(),
                "Database", seed);
        List<Integer> topology = Arrays.asList(database.getVectorSize(), 2, database.getTargetSize());
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(10000);
        manager.validateNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

    @Test
    public void or() {
        long seed = 0;
        NewDataItem[][] trainingSet = new NewDataItem[2][3];
        trainingSet[0] = createAnd();
        trainingSet[1] = createNotAnd();

        NewDataItem[][] validationSet = new NewDataItem[2][3];
        validationSet[0] = createAnd();
        validationSet[1] = createNotAnd();

        NewDataItem[][] testingSet = new NewDataItem[2][3];
        testingSet[0] = createAnd();
        testingSet[1] = createNotAnd();
        NewDatabase database = new ClassificationDatabase(trainingSet, validationSet, testingSet, new SigmoidFunction(),
                "Database", seed);
        List<Integer> topology = Arrays.asList(database.getVectorSize(), 2, database.getTargetSize());
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(10000);
        manager.validateNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

    @Test
    public void xor() {
        long seed = 0;
        NewDataItem[][] trainingSet = new NewDataItem[2][3];
        trainingSet[0] = createAnd();
        trainingSet[1] = createNotAnd();

        NewDataItem[][] validationSet = new NewDataItem[2][3];
        validationSet[0] = createAnd();
        validationSet[1] = createNotAnd();

        NewDataItem[][] testingSet = new NewDataItem[2][3];
        testingSet[0] = createAnd();
        testingSet[1] = createNotAnd();
        NewDatabase database = new ClassificationDatabase(trainingSet, validationSet, testingSet, new SigmoidFunction(),
                "Database", seed);
        List<Integer> topology = Arrays.asList(database.getVectorSize(), 3, database.getTargetSize());
        FeedForwardHeuristic heuristic = new FeedForwardHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FFClassificationNetworkManager(network, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);
        manager.trainNetwork(10000);
        manager.validateNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

}
