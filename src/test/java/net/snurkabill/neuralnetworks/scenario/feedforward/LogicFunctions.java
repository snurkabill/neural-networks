package net.snurkabill.neuralnetworks.scenario.feedforward;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.assertEquals;

public class LogicFunctions {

    @Test
    public void and() {
        long seed = 0;
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
        testingSet.put(1, createNoise());
        Database database = new Database(0, trainingSet, testingSet, "Database");
        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 2, database.getNumberOfClasses());
        FFNNHeuristic heuristic = new FFNNHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);
        manager.supervisedTraining(10000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
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


}
