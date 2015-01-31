package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.FFNNHeuristic;
import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.BinaryDeepBeliefNetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.GreedyLayerWiseStackedDeepRBMTrainer;
import net.snurkabill.neuralnetworks.managers.feedforward.FeedForwardNetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.BinaryDeepBeliefNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.ParametrizedHyperbolicTangens;
import net.snurkabill.neuralnetworks.weights.weightfactory.FineTuningWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.backpropagative.impl.online.OnlineFeedForwardNetwork;
import net.snurkabill.neuralnetworks.weights.weightfactory.LoadWeightsFactory;
import net.snurkabill.neuralnetworks.weights.weightfactory.SmartGaussianRndWeightsFactory;

public class DeepBeliefNetworkConcept extends MnistExampleFFNN {

	public static void deepBoltzmannMachine() {
		long seed = 0;
        double weightsScale = 0.0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");

        List<Integer> topology = new ArrayList<>();
        topology.add(database.getSizeOfVector());
        topology.add(500);
        topology.add(200);
        topology.add(database.getNumberOfClasses());

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();
        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", topology.get(0), topology.get(1),
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
        rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", topology.get(1), topology.get(2),
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
        List<RestrictedBoltzmannMachine> rbms2 = new ArrayList<>();
        rbms2.add(new BinaryRestrictedBoltzmannMachine("1. level", topology.get(0), topology.get(1),
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
        rbms2.add(new BinaryRestrictedBoltzmannMachine("1. level", topology.get(1), topology.get(2),
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));
		GreedyLayerWiseStackedDeepRBMTrainer.train(rbms, Arrays.asList(10000, 10000), database);
		GreedyLayerWiseStackedDeepRBMTrainer.train(rbms2, Arrays.asList(10000, 10000), database);

        DeepBoltzmannMachine dbm = new DeepBoltzmannMachine("dbm", rbms);
        DeepBoltzmannMachine dbm2 = new DeepBoltzmannMachine("dbm", rbms2);

		OnlineFeedForwardNetwork fineTuning = new OnlineFeedForwardNetwork("GreedyLayerWisePretrained-sigm", topology,
                new FineTuningWeightsFactory(dbm, topology.get(topology.size() - 1), seed, dbm.getTransferFunction()),
                FFNNHeuristic.createDefaultHeuristic(), dbm.getTransferFunction());
        NetworkManager manager = new FeedForwardNetworkManager(fineTuning, database, null);

        OnlineFeedForwardNetwork fineTuning2 = new OnlineFeedForwardNetwork("GreedyLayerWisePretrained-paramTanh", topology,
                new FineTuningWeightsFactory(dbm2, topology.get(topology.size() - 1), seed, new ParametrizedHyperbolicTangens()),
                FFNNHeuristic.createDefaultHeuristic(), new ParametrizedHyperbolicTangens());
        NetworkManager manager2 = new FeedForwardNetworkManager(fineTuning2, database, null);

        MasterNetworkManager master = new MasterNetworkManager("fine tunning", Arrays.asList(manager, manager2));
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
