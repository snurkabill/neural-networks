package net.snurkabill.neuralnetworks.examples.RBMProofOfConcept;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible.GaussianToBinaryRBM;
import net.snurkabill.neuralnetworks.utilities.datagenerator.DataGenerator;
import net.snurkabill.neuralnetworks.utilities.datagenerator.GaussianGenerator;
import net.snurkabill.neuralnetworks.utilities.datagenerator.NGaussianDistribution;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class GaussianVisibleTest {

    private static final Logger LOGGER = LoggerFactory.getLogger("GaussianVisibleTest");

    public static void gaussianVisibleExample() {

        GaussianGenerator generator1 = new GaussianGenerator(1, 0);

        DataGenerator dataGenerator = new DataGenerator(
                Arrays.asList(
                        new NGaussianDistribution(
                                Arrays.asList(
                                        generator1, generator1
                                )
                        )));

        /**
         * range:
         * [-1; 1]
         * [-1; 1]
         */

        Map<Integer, List<?>> trainingSet = new HashMap<>();
        Map<Integer, List<?>> testingSet = new HashMap<>();
        List<List<DataItem>> data = new ArrayList<>();
        List<List<DataItem>> data2 = new ArrayList<>();

            data.add(new ArrayList<DataItem>());
            data2.add(new ArrayList<DataItem>());
            for (int j = 0; j < 1_000_000; j++) {
                data.get(0).add(dataGenerator.generateNext());
                data2.get(0).add(dataGenerator.generateNext());
            }

        for (int i = 0; i < data.size(); i++) {
            trainingSet.put(i, data.get(i));
            testingSet.put(i, data2.get(i));
        }
        Database database = new Database(0, trainingSet, testingSet, "testingDatabaseRBM", true);

        BoltzmannMachineHeuristic heuristics = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristics.batchSize = 1000;
        heuristics.learningRate = 0.001;

        GaussianToBinaryRBM rbm = new GaussianToBinaryRBM(
                "RBMTest", database.getSizeOfVector(), 100,
                new GaussianRndWeightsFactory(0.0, 0), heuristics, 0);

        UnsupervisedRBMManager manager = new UnsupervisedRBMManager(rbm, database, 0, null,
                new ProbabilisticAssociationVectorValidator(1));
        for (int i = 0; i < 100; i++) {
            manager.supervisedTraining(10000);
            manager.testNetwork();
            LOGGER.info("Error: {}", manager.getTestResults().getComparableSuccess());
        }
        Database.InfiniteRandomTrainingIterator iterator = database.getInfiniteRandomTrainingIterator();
        for (int i = 0; i < 100; i++) {
            DataItem item = iterator.next();

            rbm.calculateNetwork(item.data);
            LOGGER.info("Original: {}, Reconstruction: {} ", item.data, rbm.getVisibleNeurons());
        }
    }
}
