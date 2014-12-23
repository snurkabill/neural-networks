package net.snurkabill.neuralnetworks.scenario.deep;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.BinaryDeepBeliefNetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.StuckedRBMTrainer;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.BinaryDeepBeliefNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.scenario.LogicFunctions;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.junit.Test;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;

public class DBNProofOfConceptTest extends LogicFunctions {

    @Test
    public void and() {
        long seed = 0;
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();
        trainingSet.put(0, createAnd());
        trainingSet.put(1, createNotAnd());
        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 10000; i++) {
            and.addAll(createAnd());
            noise.addAll(createNotAnd());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);
        Database database = new Database(seed, trainingSet, testingSet, "Database");

        double weightsScale = 0.1;

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();
        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), 100,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
        rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", 100, 100,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
        StuckedRBM inputTransfer = new StuckedRBM(rbms, "NetworkChimney");
        StuckedRBMTrainer.train(inputTransfer, database, 100, 100);
        BinaryDeepBeliefNetwork theBeast = new BinaryDeepBeliefNetwork("theBeast", database.getNumberOfClasses(),
                100, new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicDBN(), seed, inputTransfer);
        NetworkManager manager = new BinaryDeepBeliefNetworkManager(theBeast, database,
                null);
        manager.supervisedTraining(1000);
        manager.testNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);

    }

}
