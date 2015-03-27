package net.snurkabill.neuralnetworks.examples.RBMProofOfConcept;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.heuristic.FeedForwardHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.binaryvisible.BinaryToBinaryRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.DeepOnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.weights.weightfactory.FineTuningWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

public class RBMTest {

    private static final Logger LOGGER = LoggerFactory.getLogger("RBM test");

    public void test() {

        long seed = 0;
        LOGGER.info("NEXT TEST");

        List<double[]> vectors = new ArrayList<>();
        //vectors.add(new double[] {1, 1, 1, 1, 1, 0, 0, 0, 0, 0});
        vectors.add(new double[]{1, 0});
        vectors.add(new double[]{0, 1});

        Map<Integer, List<?>> trainingSet = new HashMap<>();
        Map<Integer, List<?>> testingSet = new HashMap<>();
        List<List<DataItem>> data = new ArrayList<>();
        List<List<DataItem>> data2 = new ArrayList<>();

        for (int i = 0; i < vectors.size(); i++) {
            int multiplicator = 100;
            data.add(new ArrayList<DataItem>());
            data2.add(new ArrayList<DataItem>());
            for (int j = 0; j < 100; j++) {
                data.get(i).add(new DataItem(vectors.get(i)));
                data2.get(i).add(new DataItem(vectors.get(i)));
            }
        }
        for (int i = 0; i < data.size(); i++) {
            trainingSet.put(i, data.get(i));
            testingSet.put(i, data2.get(i));
        }
        Database database = new Database(0, trainingSet, testingSet, "testingDatabaseRBM", true);

        List<Integer> topology = Arrays.asList(database.getSizeOfVector(), 1, database.getNumberOfClasses());

        BoltzmannMachineHeuristic heuristics = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristics.batchSize = 3;
        heuristics.learningRate = 0.01;

        BinaryToBinaryRBM rbm = new BinaryToBinaryRBM(
                "RBMTest", topology.get(0), topology.get(1),
                new GaussianRndWeightsFactory(0.0, 0), heuristics, 0);

        /*BinaryRestrictedBoltzmannMachine rbm2 = new BinaryRestrictedBoltzmannMachine(
                "RBMTest", topology.get(0), topology.get(1),
                new GaussianRndWeightsFactory(0.0, 0), heuristics, 0);*/

        UnsupervisedRBMManager manager = new UnsupervisedRBMManager(rbm, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10));
        /*UnsupervisedRBMManager manager2 = new UnsupervisedRBMManager(rbm2, database, 0, null,
                new ProbabilisticAssociationVectorValidator(10));*/
        for (int i = 0; i < 100; i++) {
            manager.supervisedTraining(1000);
            //  manager2.supervisedTraining(1000);
            //  manager.testNetwork();
            //  manager2.testNetwork();
//            LOGGER.info("Error: {}", manager.getTestResults().getComparableSuccess());
        }
        /*for (int i = 0; i < 100; i++) {
            LOGGER.info("Reconstructed: {}", rbm.reconstructNext());
        }*/

        DeepBoltzmannMachine dbm = new DeepBoltzmannMachine("dbm", Arrays.asList((RestrictedBoltzmannMachine) rbm));
        //DeepBoltzmannMachine dbm2 = new DeepBoltzmannMachine("dbm", Arrays.asList((RestrictedBoltzmannMachine)rbm2));

        FeedForwardHeuristic linearHeuristics = FeedForwardHeuristic.createDefaultHeuristic();
        linearHeuristics.learningRate = 0.01;
        DeepOnlineFeedForwardNetwork lastLayerTrainer = new DeepOnlineFeedForwardNetwork("deep",
                Arrays.asList(topology.get(1), topology.get(2)),
                new SmartGaussianRndWeightsFactory(new SigmoidFunction(), seed), linearHeuristics,
                new ParametrizedHyperbolicTangens(), dbm, 10);
        NetworkManager linearManager = new FeedForwardNetworkManager(lastLayerTrainer, database, null);
        linearManager.supervisedTraining(100_00);
        linearManager.testNetwork();
        LOGGER.info("Deep online results: {}", linearManager.getTestResults().getComparableSuccess());

        OnlineFeedForwardNetwork fineTuning = new OnlineFeedForwardNetwork("pretrained linear classifier", topology,
                new FineTuningWeightsFactory(dbm, lastLayerTrainer),
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager3 = new FeedForwardNetworkManager(fineTuning, database, null);

        /*OnlineFeedForwardNetwork fineTuning2 = new OnlineFeedForwardNetwork("training whole machinery", topology,
                new FineTuningWeightsFactory(dbm2, database.getNumberOfClasses(), seed, 0.001),
                FFNNHeuristic.createDefaultHeuristic(), new SigmoidFunction());
        NetworkManager manager4 = new FeedForwardNetworkManager(fineTuning2, database, null);*/

        OnlineFeedForwardNetwork network = new OnlineFeedForwardNetwork("SmartParametrizedTanh", topology,
                new SmartGaussianRndWeightsFactory(new ParametrizedHyperbolicTangens(), seed),
                FeedForwardHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager5 = new FeedForwardNetworkManager(network, database, null);

        MasterNetworkManager master = new MasterNetworkManager(
                "fine tunning", Arrays.asList(manager3, /*manager4,*/ manager5), 2);
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(1000, 1, master);
        benchmarker.benchmark();
    }
}

