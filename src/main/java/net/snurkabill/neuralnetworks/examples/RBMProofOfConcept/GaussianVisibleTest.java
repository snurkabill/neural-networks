package net.snurkabill.neuralnetworks.examples.RBMProofOfConcept;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.ClassFullDatabase;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible.GaussianToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible.GaussianToRectifiedRBM;
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
        ClassFullDatabase classFullDatabase = new ClassFullDatabase(0, trainingSet, testingSet, "testingDatabaseRBM", true);

        BoltzmannMachineHeuristic heuristics = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristics.batchSize = 1;
        heuristics.learningRate = 0.000001;
        heuristics.constructiveDivergenceIndex = 5;
        heuristics.momentum = 0.3;

        GaussianToBinaryRBM rbm = new GaussianToRectifiedRBM(
                "RBM-Rectified", classFullDatabase.getSizeOfVector(), 30,
                new GaussianRndWeightsFactory(1, 0), heuristics, 0);

        UnsupervisedRBMManager manager = new UnsupervisedRBMManager(rbm, classFullDatabase, 0, null,
                new ProbabilisticAssociationVectorValidator(1));

        GaussianToBinaryRBM rbm2 = new GaussianToBinaryRBM(
                "RBM-Binary", classFullDatabase.getSizeOfVector(), 30,
                new GaussianRndWeightsFactory(1, 0), heuristics, 0);

        UnsupervisedRBMManager manager2 = new UnsupervisedRBMManager(rbm2, classFullDatabase, 0, null,
                new ProbabilisticAssociationVectorValidator(1));

        MasterNetworkManager master = new MasterNetworkManager("2D-gauss", Arrays.<NetworkManager>asList(manager, manager2), 2);

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(100, 10_000, master);
        benchmarker.benchmark();
        /*for (int i = 0; i < 100; i++) {
            manager.supervisedTraining(10000);
            manager.testNetwork();
            LOGGER.info("Error: {}", manager.getTestResults().getComparableSuccess());
        }
        ClassFullDatabase.InfiniteRandomTrainingIterator iterator = classFullDatabase.getInfiniteRandomTrainingIterator();
        for (int i = 0; i < 100; i++) {
            DataItem item = iterator.next();

            rbm.calculateNetwork(item.data);
            LOGGER.info("Original: {}, Reconstruction: {} ", item.data, rbm.getVisibleNeurons());
        }*/
    }
}
