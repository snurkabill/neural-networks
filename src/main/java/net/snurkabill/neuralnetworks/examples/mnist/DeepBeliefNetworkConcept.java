package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.deep.GreedyLayerWiseStackedDeepRBMTrainer;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.DeepOnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.weights.weightfactory.FineTuningWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;

import java.util.*;

public class DeepBeliefNetworkConcept extends MnistExampleFFNN {

    protected static List<DataItem> createXor() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{0.0, 1.0}));
        list.add(new DataItem(new double[]{1.0, 0.0}));
        return list;
    }

    protected static List<DataItem> createNotXor() {
        List<DataItem> list = new ArrayList<>();
        list.add(new DataItem(new double[]{1.0, 1.0}));
        list.add(new DataItem(new double[]{0.0, 0.0}));
        return list;
    }

    public static void deepXor() {
        Map<Integer, List<DataItem>> trainingSet = new HashMap<>();
        Map<Integer, List<DataItem>> testingSet = new HashMap<>();
        List<DataItem> and = new ArrayList<>();
        List<DataItem> noise = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            and.addAll(createXor());
            noise.addAll(createNotXor());
        }
        trainingSet.put(0, and);
        trainingSet.put(1, noise);
        and = new ArrayList<>();
        noise = new ArrayList<>();
        for (int i = 0; i < 1; i++) {
            and.addAll(createXor());
            noise.addAll(createNotXor());
        }
        testingSet.put(0, and);
        testingSet.put(1, noise);
        Database database = new Database(0, trainingSet, testingSet, "Database", true);
        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 8, database.getNumberOfClasses());

        HeuristicRBM heuristicRbm = new HeuristicRBM();
        heuristicRbm.batchSize = 2;
        heuristicRbm.learningRate = 0.1;
        heuristicRbm.momentum = 0.0;
        heuristicRbm.temperature = 1;
        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();
        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), topology.get(1),
                new GaussianRndWeightsFactory(100, 0), heuristicRbm, 0));
        List<RestrictedBoltzmannMachine> rbms2 = new ArrayList<>();
        rbms2.add(new BinaryRestrictedBoltzmannMachine("2. level", database.getSizeOfVector(), topology.get(1),
                new GaussianRndWeightsFactory(100, 0), heuristicRbm, 0));

        DeepBoltzmannMachine dbm = new DeepBoltzmannMachine("dbm1", rbms);
        DeepBoltzmannMachine dbm2 = new DeepBoltzmannMachine("dbm2", rbms2);

        /*NetworkManager rbmManager = new UnsupervisedRBMManager(rbms.get(0), database, 0, null,
                new ProbabilisticAssociationVectorValidator(1000));
*/
        //GreedyLayerWiseStackedDeepRBMTrainer.train(rbms, Arrays.asList(100000), database);

        FFNNHeuristic heuristic = new FFNNHeuristic();
        heuristic.learningRate = 0.01;
        heuristic.momentum = 0.0;

        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("xor-networkSigm", topology,
                new GaussianRndWeightsFactory(10, 0), heuristic,
                new SigmoidFunction());
        NetworkManager manager = new FeedForwardNetworkManager(network, database, null);
        OnlineFeedForwardNetwork network2 = new OnlineFeedForwardNetwork("xor-networkTanh", topology,
                new GaussianRndWeightsFactory(10, 0), heuristic,
                new ParametrizedHyperbolicTangens());
        NetworkManager manager2 = new FeedForwardNetworkManager(network2, database, null);


        DeepOnlineFeedForwardNetwork lastLayerTrainer = new DeepOnlineFeedForwardNetwork("PretrainedLinearClassifier",
                Arrays.asList(topology.get(1), topology.get(2)),
                new GaussianRndWeightsFactory(0.001, 0), FFNNHeuristic.createDefaultHeuristic(),
                new ParametrizedHyperbolicTangens(), dbm, 10);
        NetworkManager manager3 = new FeedForwardNetworkManager(lastLayerTrainer, database, null);
        manager.supervisedTraining(1000);
        manager.testNetwork();
        LOGGER.info("Deep online results: {}", manager.getTestResults().getComparableSuccess());

        OnlineFeedForwardNetwork fineTuning = new OnlineFeedForwardNetwork("pretrained linear classifier", topology,
                new FineTuningWeightsFactory(dbm, lastLayerTrainer),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager4 = new FeedForwardNetworkManager(fineTuning, database, null);

        OnlineFeedForwardNetwork fineTuning2 = new OnlineFeedForwardNetwork("training whole machinery", topology,
                new FineTuningWeightsFactory(dbm2, database.getNumberOfClasses(), 0, 0.001),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager5 = new FeedForwardNetworkManager(fineTuning2, database, null);

        OnlineFeedForwardNetwork network10 = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), 0),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager6 = new FeedForwardNetworkManager(network10, database, null);


        MasterNetworkManager master = new MasterNetworkManager("xor problem", Arrays.asList(manager, manager2,
                manager3, manager4, manager5, manager6));
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(100, 100, master);
        benchmarker.benchmark();
    }

    public static void errorFunctionOnRBMs() {
        long seed = 10;
        double weightsScale = 0.0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

        HeuristicRBM heuristicRbm = new HeuristicRBM();
        heuristicRbm.batchSize = 10;
        heuristicRbm.learningRate = 0.01;
        heuristicRbm.momentum = 0.01;

        NetworkManager manager1 = new UnsupervisedRBMManager(
                (new BinaryRestrictedBoltzmannMachine("RBM1", database.getSizeOfVector(), 10,
                        new GaussianRndWeightsFactory(1, 0), heuristicRbm, 0)), database, seed, null,
                new ProbabilisticAssociationVectorValidator(5));
        NetworkManager manager2 = new UnsupervisedRBMManager(
                (new BinaryRestrictedBoltzmannMachine("RBM2", database.getSizeOfVector(), 50,
                        new GaussianRndWeightsFactory(1, 0), heuristicRbm, 0)), database, seed, null,
                new ProbabilisticAssociationVectorValidator(5));
        NetworkManager manager3 = new UnsupervisedRBMManager(
                (new BinaryRestrictedBoltzmannMachine("RBM3", database.getSizeOfVector(), 100,
                        new GaussianRndWeightsFactory(1, 0), heuristicRbm, 0)), database, seed, null,
                new ProbabilisticAssociationVectorValidator(5));
        NetworkManager manager4 = new UnsupervisedRBMManager(
                (new BinaryRestrictedBoltzmannMachine("RBM4", database.getSizeOfVector(), 200,
                        new GaussianRndWeightsFactory(1, 0), heuristicRbm, 0)), database, seed, null,
                new ProbabilisticAssociationVectorValidator(5));

        MasterNetworkManager master = new MasterNetworkManager("errorOnRBMs", Arrays.asList(manager1, manager2,
                manager3, manager4));
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(10, 1000, master);
        benchmarker.benchmark();
    }

    public static void deepBoltzmannMachine() {
        long seed = 0;
        double weightsScale = 0.01;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

        List<Integer> topology = new ArrayList<>();
        topology.add(database.getSizeOfVector());
        topology.add(100);
        topology.add(database.getNumberOfClasses());

        HeuristicRBM heuristic = new HeuristicRBM();
        heuristic.batchSize = 10;
        heuristic.learningRate = 0.3;
        heuristic.momentum = 0.1;
        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();
        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", topology.get(0), topology.get(1),
                new GaussianRndWeightsFactory(0, seed), heuristic, seed));

        GreedyLayerWiseStackedDeepRBMTrainer.train(rbms, Arrays.asList(1000), database);

        DeepBoltzmannMachine dbm = new DeepBoltzmannMachine("dbm", rbms);

        FFNNHeuristic linearHeuristics = FFNNHeuristic.createDefaultHeuristic();
        linearHeuristics.learningRate = 0.01;
        DeepOnlineFeedForwardNetwork lastLayerTrainer = new DeepOnlineFeedForwardNetwork("deep",
                Arrays.asList(topology.get(1), topology.get(2)),
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),linearHeuristics,
                new ParametrizedHyperbolicTangens(), dbm, 10);
        NetworkManager manager = new FeedForwardNetworkManager(lastLayerTrainer, database, null);
        manager.supervisedTraining(1000);
        manager.testNetwork();
        LOGGER.info("Deep online results: {}", manager.getTestResults().getComparableSuccess());

        OnlineFeedForwardNetwork fineTuning = new OnlineFeedForwardNetwork("pretrained linear classifier", topology,
                new FineTuningWeightsFactory(dbm, lastLayerTrainer),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager3 = new FeedForwardNetworkManager(fineTuning, database, null);

        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager2 = new FeedForwardNetworkManager(network, database, null);

        /*OnlineFeedForwardNetwork network2 = new OnlineFeedForwardNetwork("SmartParametrizedTanh2", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FFNNHeuristic.littleStepsWithHugeMoment(), new ParametrizedHyperbolicTangens());
        NetworkManager manager4 = new FeedForwardNetworkManager(network2, database, null);*/

        MasterNetworkManager master = new MasterNetworkManager("fine tunning", Arrays.asList(manager3, manager2/*, manager4*/));
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(10, 1000, master);
        benchmarker.benchmark();
    }

    /*public static void showMePowerOfDBN() {
        long seed = 0;
        double weightsScale = 0.001;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();

        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), 500,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));

        rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", 500, 500,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));

        GreedyLayerWiseStackedDeepRBMTrainer inputTransfer = new GreedyLayerWiseStackedDeepRBMTrainer(rbms, "NetworkChimney");

        net.snurkabill.neuralnetworks.managers.deep.StuckedRBMTrainer.train(inputTransfer, database, 2000, 2000);

        BinaryDeepBeliefNetwork theBeast = new BinaryDeepBeliefNetwork("theBeast", database.getNumberOfClasses(),
                1000, new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicDBN(), seed, inputTransfer);

        NetworkManager manager = new BinaryDeepBeliefNetworkManager(theBeast, database,
                null);

        MasterNetworkManager master = new MasterNetworkManager("dbn testing",
                Collections.singletonList(manager));

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(10, 1000, master);
        benchmarker.benchmark();


    }*/

}
