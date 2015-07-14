package net.snurkabill.neuralnetworks.scenario.energybased;

import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.RBMClassificationNetworkManager;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.scenario.LogicFunctions;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;

public class RBMProofOfConceptTest extends LogicFunctions {

    private static final Logger LOGGER = LoggerFactory.getLogger(RBMProofOfConceptTest.class);

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
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.0;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 1;

        BinaryToBinaryRBM machine = new BinaryToBinaryRBM("AndRBM", database.getVectorSize() + database.getTargetSize(),
                100, new GaussianRndWeightsFactory(0, seed), heuristic, seed);

        RBMClassificationNetworkManager manager = new RBMClassificationNetworkManager(machine, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        // declarative way
        for (int i = 0; i < 10000; i++) {
            manager.trainNetwork(1000);
            manager.validateNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 95) {
                break;
            }
        }
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
        BoltzmannMachineHeuristic heuristic = new BoltzmannMachineHeuristic();
        heuristic.learningRate = 0.1;
        heuristic.momentum = 0.0;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 1;

        BinaryToBinaryRBM machine = new BinaryToBinaryRBM("OrRBM", database.getVectorSize() + database.getTargetSize(),
                100, new GaussianRndWeightsFactory(0, seed), heuristic, seed);

        RBMClassificationNetworkManager manager = new RBMClassificationNetworkManager(machine, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);


        // declarative way
        for (int i = 0; i < 10000; i++) {
            manager.validateNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 90) {
                break;
            }
            manager.trainNetwork(1000);
        }
        manager.validateNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }

    /*@Test
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
            manager.validateNetwork();
            LOGGER.debug("Success: {}, Iteration {}", manager.getTestResults().getComparableSuccess(), i);
            if(manager.getTestResults().getComparableSuccess() > 90) {
                break;
            }
            manager.trainNetwork(10);
        }
        manager.validateNetwork();
        assertEquals("Result:" + manager.getTestResults().getComparableSuccess(),
                100.0, manager.getTestResults().getComparableSuccess(), 0.000001);
    }*/
}
