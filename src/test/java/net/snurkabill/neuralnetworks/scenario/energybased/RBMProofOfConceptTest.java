package net.snurkabill.neuralnetworks.scenario.energybased;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.scenario.LogicFunctions;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class RBMProofOfConceptTest extends LogicFunctions {

    private static final Logger LOGGER = LoggerFactory.getLogger(RBMProofOfConceptTest.class);

    @Test
    public void and() {
        long seed = 0;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();
        trainingSet.put(0, createAnd());
        trainingSet.put(1, createNotAnd());
        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            and.addAll(createAnd());
            noise.addAll(createNotAnd());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);
        Database database = new Database(0, trainingSet, testingSet, "Database", true);
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.learningRate = 0.001;
        heuristic.momentum = 0.1;
        heuristic.constructiveDivergenceIndex = 100;
        heuristic.temperature = 0.1;
        heuristic.batchSize = 6;

        BinaryToBinaryRBM machine = new BinaryToBinaryRBM("And-RBM", database.getSizeOfVector() + database.getNumberOfClasses(),
                10, new GaussianRndWeightsFactory(0, seed), heuristic, seed);

        SupervisedRBMManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        // declarative way
        for (int i = 0; i < 10000; i++) {
            manager.testNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 90) {
                break;
            }
            manager.supervisedTraining(1000);
        }
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

    @Test
    public void or() {
        long seed = 0;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();
        trainingSet.put(0, createOr());
        trainingSet.put(1, createNotOr());
        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            and.addAll(createOr());
            noise.addAll(createNotOr());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);
        Database database = new Database(0, trainingSet, testingSet, "Database", true);
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.learningRate = 0.001;
        heuristic.momentum = 0.1;
        heuristic.constructiveDivergenceIndex = 100;
        heuristic.temperature = 0.1;
        heuristic.batchSize = 6;

        BinaryToBinaryRBM machine = new BinaryToBinaryRBM("Or-RBM", database.getSizeOfVector() + database.getNumberOfClasses(),
                10, new GaussianRndWeightsFactory(0, seed), heuristic, seed);

        SupervisedRBMManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        // declarative way
        for (int i = 0; i < 10000; i++) {
            manager.testNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 90) {
                break;
            }
            manager.supervisedTraining(1000);
        }
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

    @Test
    @Ignore
    public void xor() {
        long seed = 0;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();
        trainingSet.put(0, createXor());
        trainingSet.put(1, createNotXor());
        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 10; i++) {
            and.addAll(createXor());
            noise.addAll(createNotXor());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);
        Database database = new Database(0, trainingSet, testingSet, "Database", true);
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.learningRate = 0.001;
        heuristic.momentum = 0.1;
        heuristic.constructiveDivergenceIndex = 100;
        heuristic.temperature = 0.1;
        heuristic.batchSize = 6;

        BinaryToBinaryRBM machine = new BinaryToBinaryRBM("And-RBM", database.getSizeOfVector() + database.getNumberOfClasses(),
                100, new GaussianRndWeightsFactory(0, seed), heuristic, seed);

        SupervisedRBMManager manager = new SupervisedRBMManager(machine, database, seed, null,
                new PartialProbabilisticAssociationVectorValidator(1, database.getNumberOfClasses()));

        // declarative way
        for (int i = 0; i < 10000; i++) {
            manager.testNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 90) {
                break;
            }
            manager.supervisedTraining(10);
        }
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }
}
