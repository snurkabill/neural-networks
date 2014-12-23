package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.heuristic.HeuristicRBM;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.BinaryDeepBeliefNetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.StuckedRBMTrainer;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.BinaryDeepBeliefNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class DeepBeliefNetworkConcept extends MnistExampleFFNN {

    public static void showMePowerOfDBN() {
        long seed = 0;
        double weightsScale = 0.001;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST");

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();

        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), 500,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));

        rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", 500, 500,
                new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicRBM(), seed));

        StuckedRBM inputTransfer = new StuckedRBM(rbms, "NetworkChimney");

        StuckedRBMTrainer.train(inputTransfer, database, 2000, 2000);

        BinaryDeepBeliefNetwork theBeast = new BinaryDeepBeliefNetwork("theBeast", database.getNumberOfClasses(),
                1000, new GaussianRndWeightsFactory(weightsScale, seed), new HeuristicDBN(), seed, inputTransfer);

        NetworkManager manager = new BinaryDeepBeliefNetworkManager(theBeast, database,
                null);

        MasterNetworkManager master = new MasterNetworkManager("dbn testing",
                Collections.singletonList(manager));

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(10, 1000, master);
        benchmarker.benchmark();


    }

}
