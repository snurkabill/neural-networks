package net.snurkabill.neuralnetworks.scenario.energybased;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.scenario.LogicFunctions;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.junit.Ignore;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class RBMProofOfConceptTest extends LogicFunctions {

    @Ignore
    @Test
    public void and() {
        long seed = 0;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();

        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 1.0}));

        trainingSet.put(0, list);

        List<DataItem> and = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            and.add(new DataItem(new double[]{1.0, 1.0}));
            and.add(new DataItem(new double[]{0.0, 0.0}));
            and.add(new DataItem(new double[]{1.0, 0.0}));
            and.add(new DataItem(new double[]{0.0, 1.0}));
        }
        testingSet.put(0, and);

        Database database = new Database(0, trainingSet, testingSet, "Database");

        List<NetworkManager> managers = new ArrayList<>();

        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.batchSize = 1;
        heuristic.constructiveDivergenceIndex = 1;
        BinaryRestrictedBoltzmannMachine machine = new BinaryRestrictedBoltzmannMachine("RBM1",
                database.getSizeOfVector(), 10,
                new GaussianRndWeightsFactory(0.01, seed), heuristic, seed);
        managers.add(new UnsupervisedRBMManager(machine, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10)));

        heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.batchSize = 3;
        heuristic.constructiveDivergenceIndex = 1;
        machine = new BinaryRestrictedBoltzmannMachine("RBM2", database.getSizeOfVector(), 10,
                new GaussianRndWeightsFactory(0.01, seed), heuristic, seed);
        managers.add(new UnsupervisedRBMManager(machine, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10)));

        heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.batchSize = 10;
        heuristic.constructiveDivergenceIndex = 1;
        machine = new BinaryRestrictedBoltzmannMachine("RBM3", database.getSizeOfVector(), 10,
                new GaussianRndWeightsFactory(0.01, seed), heuristic, seed);
        managers.add(new UnsupervisedRBMManager(machine, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10)));

        heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.batchSize = 50;
        heuristic.constructiveDivergenceIndex = 1;
        machine = new BinaryRestrictedBoltzmannMachine("RBM4", database.getSizeOfVector(), 10,
                new GaussianRndWeightsFactory(0.01, seed), heuristic, seed);
        managers.add(new UnsupervisedRBMManager(machine, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10)));

        heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.batchSize = 100;
        heuristic.constructiveDivergenceIndex = 1;
        machine = new BinaryRestrictedBoltzmannMachine("RBM5", database.getSizeOfVector(), 10,
                new GaussianRndWeightsFactory(0.01, seed), heuristic, seed);
        managers.add(new UnsupervisedRBMManager(machine, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10)));

    }
}
