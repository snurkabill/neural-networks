package net.snurkabill.neuralnetworks.examples.RBMProofOfConcept;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.unsupervised.UnsupervisedNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible.GaussianToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.gaussianvisible.GaussianToRectifiedRBM;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.newdata.database.RegressionDatabase;
import net.snurkabill.neuralnetworks.utilities.datagenerator.DataGenerator;
import net.snurkabill.neuralnetworks.utilities.datagenerator.GaussianGenerator;
import net.snurkabill.neuralnetworks.utilities.datagenerator.NGaussianDistribution;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

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
        NewDataItem[] trainingSet = new NewDataItem[100_000];
        NewDataItem[] testingSet = new NewDataItem[100_000];
        NewDataItem[] validationSet = new NewDataItem[100_000];

        for (int j = 0; j < 1_000_000; j++) {
            trainingSet[j] = dataGenerator.generateNext();
            testingSet[j] = dataGenerator.generateNext();
            validationSet[j] = dataGenerator.generateNext();
        }
        NewDatabase database = new RegressionDatabase(trainingSet, validationSet, testingSet, "testingDatabaseRBM", 0);

        BoltzmannMachineHeuristic heuristics = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristics.batchSize = 1;
        heuristics.learningRate = 0.000001;
        heuristics.constructiveDivergenceIndex = 5;
        heuristics.momentum = 0.3;

        GaussianToBinaryRBM rbm = new GaussianToRectifiedRBM(
                "RBM-Rectified", database.getVectorSize(), 30,
                new GaussianRndWeightsFactory(1, 0), heuristics, 0);

        UnsupervisedNetworkManager manager = new UnsupervisedNetworkManager(rbm, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        GaussianToBinaryRBM rbm2 = new GaussianToBinaryRBM(
                "RBM-Binary", database.getVectorSize(), 30,
                new GaussianRndWeightsFactory(1, 0), heuristics, 0);

        UnsupervisedNetworkManager manager2 = new UnsupervisedNetworkManager(rbm2, database, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        MasterNetworkManager master = new MasterNetworkManager("2D-gauss", Arrays.<NetworkManager>asList(manager, manager2), 2);

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(100, 10_000, master, "rbmGaussianVisible");
        benchmarker.benchmark();
        /*for (int i = 0; i < 100; i++) {
            manager.trainNetwork(10000);
            manager.validateNetwork();
            LOGGER.info("Error: {}", manager.getTestResults().getComparableSuccess());
        }
        Database.InfiniteRandomTrainingIterator iterator = database.getInfiniteRandomTrainingIterator();
        for (int i = 0; i < 100; i++) {
            DataItem item = iterator.next();

            rbm.calculateNetwork(item.data);
            LOGGER.info("Original: {}, Reconstruction: {} ", item.data, rbm.getVisibleNeurons());
        }*/
    }
}
