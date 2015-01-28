package net.snurkabill.neuralnetworks.examples.gaussianvisible;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.SupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.PartialProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.ContinuousRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.GaussianVisibleRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

/**
 * does not work :( :D
 */
public class GaussianVisibleImplementation {

    public static Logger LOGGER = LoggerFactory.getLogger("GaussianRBMExample");

    public static void proofOfConcept() {

        DataItem sample1 = new DataItem(new double[3]);
        DataItem sample2 = new DataItem(new double[3]);
        DataItem sample3 = new DataItem(new double[3]);
        DataItem sample4 = new DataItem(new double[3]);

        sample1.data[0] = 0;
        sample1.data[1] = 0;
        sample1.data[2] = 0;

        sample2.data[0] = 1;
        sample2.data[1] = 1;
        sample2.data[2] = 1;

        sample3.data[0] = -1;
        sample3.data[1] = -1;
        sample3.data[2] = -1;

        sample4.data[0] = -1;
        sample4.data[1] = 1;
        sample4.data[2] = 10;

        List<DataItem> data = Arrays.asList(/*sample1,*/ sample2, sample3/*, sample4*/);
        List<DataItem> dataBinary = Arrays.asList(sample1, sample2);

        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.learningRate = 0.000001;
        heuristic.momentum = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 10;

        GaussianVisibleRestrictedBoltzmannMachine rbm = new GaussianVisibleRestrictedBoltzmannMachine("ProofOfConcept",
                3, 1, new GaussianRndWeightsFactory(0.001, 0), heuristic, 0);

        BinaryRestrictedBoltzmannMachine rbmBin = new BinaryRestrictedBoltzmannMachine("Binary",
                3, 5, new GaussianRndWeightsFactory(0.000001, 0), heuristic, 0);

        int testingIterations = 1000;
        double error = 0.0;
        double errorBinary = 0.0;
        for (int i = 0; i < testingIterations; i++) {
            for (int j = 0; j < data.size(); j++) {
                error += rbm.calcError(data.get(j).data, data.get(j).data);
            }
            for (int j = 0; j < dataBinary.size(); j++) {
                errorBinary += rbmBin.calcError(dataBinary.get(j).data, dataBinary.get(j).data);
            }
        }
        LOGGER.info("Error before training: {}", error);
        LOGGER.info("BinError before training: {}", errorBinary);
        int totalIterations = 10;
        for (int k = 0; k < totalIterations; k++) {
            int learningIterations = 1000;
            for (int i = 0; i < learningIterations; i++) {
                Collections.shuffle(data);
                for (DataItem aData : data) {
                    rbm.trainNetwork(aData.data);
                }
                /*Collections.shuffle(dataBinary);
                for (DataItem aDataBinary : dataBinary) {
                    rbmBin.trainMachine(aDataBinary.data);
                }*/
            }
            error = 0.0;
            errorBinary = 0.0;
            for (int i = 0; i < testingIterations; i++) {
                for (int j = 0; j < data.size(); j++) {
                    error += rbm.calcError(data.get(j).data, data.get(j).data);
                }
                for (int j = 0; j < dataBinary.size(); j++) {
                    errorBinary += rbmBin.calcError(dataBinary.get(j).data, dataBinary.get(j).data);
                }
            }
            LOGGER.info("Error during training: {} {}", k, error);
            LOGGER.info("BinError during training: {} {}", k, errorBinary);
        }

        error = 0.0;
        errorBinary = 0.0;
        for (int i = 0; i < testingIterations; i++) {
            for (int j = 0; j < data.size(); j++) {
                error += rbm.calcError(data.get(j).data, data.get(j).data);
            }
            for (int j = 0; j < dataBinary.size(); j++) {
                errorBinary += rbmBin.calcError(dataBinary.get(j).data, dataBinary.get(j).data);
            }
        }
        LOGGER.info("Error after training: {} ", error);
        LOGGER.info("BinError after training: {} ", errorBinary);

    }

    private static List<DataItem> createNotAnd() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

    private static List<DataItem> createAnd() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 1.0}));
        return list;
    }

    public static void shitFuck() {
        long seed = 0;
        double weightsScale = 0.01;
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
        Database database = new Database(0, trainingSet, testingSet, "Database", true);

        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 2, database.getNumberOfClasses());
        FFNNHeuristic heuristic = new FFNNHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.1;
        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                heuristic, new ParametrizedHyperbolicTangens());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);

        ContinuousRestrictedBoltzmannMachine rbm = new ContinuousRestrictedBoltzmannMachine("contRBM",
                database.getSizeOfVector(), 10, new GaussianRndWeightsFactory(0, 0),
                HeuristicRBM.createStartingHeuristicParams(), 0);

        NetworkManager manager2 = new UnsupervisedRBMManager(rbm, database, 0, null,
                new ProbabilisticAssociationVectorValidator(5));

        manager2.supervisedTraining(10);
        manager2.testNetwork();
        System.out.println("Results " + manager2.getTestResults().getComparableSuccess());

        manager2.supervisedTraining(10);
        manager2.testNetwork();
        System.out.println("Results " + manager2.getTestResults().getComparableSuccess());

        manager2.supervisedTraining(1000);
        manager2.testNetwork();
        System.out.println("Results " + manager2.getTestResults().getComparableSuccess());

        /*manager.supervisedTraining(10000);
        manager.testNetwork();
        System.out.println("Results " + manager.getTestResults().getComparableSuccess());*/


    }

}
