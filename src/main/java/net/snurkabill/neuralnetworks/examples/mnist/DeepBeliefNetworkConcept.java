package net.snurkabill.neuralnetworks.examples.mnist;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.heuristic.HeuristicDBN;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.UnsupervisedRBMManager;
import net.snurkabill.neuralnetworks.managers.boltzmannmodel.validator.ProbabilisticAssociationVectorValidator;
import net.snurkabill.neuralnetworks.managers.deep.BinaryDeepBeliefNetworkManager;
import net.snurkabill.neuralnetworks.managers.deep.StuckedRBMTrainer;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.BinaryDeepBeliefNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class DeepBeliefNetworkConcept extends MnistExampleFFNN {

    private static Logger log = LoggerFactory.getLogger("FIST EXAMPLE");

    public static void deepBoltzmannMachine() {
        System.out.println("DBN power");
        long seed = 101;
        double weightsScale = 0.0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();

        HeuristicDBN heuristic = new HeuristicDBN();
        heuristic.learningRate = 0.1;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 100;

        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), 100,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", 100, 100,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        /*rbms.add(new BinaryRestrictedBoltzmannMachine("3. level", 50, 50,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
*/
        DeepRestrictedBoltzmannMachine dbm = new DeepRestrictedBoltzmannMachine("dbm", 0,
                rbms, Arrays.asList(100_000, 100_000/*, 100_000)*/));

        BinaryRestrictedBoltzmannMachine rbm = new BinaryRestrictedBoltzmannMachine("rbm", database.getSizeOfVector(),
                100, new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed);

        NetworkManager manager = new UnsupervisedRBMManager(dbm, database, 0, null,
                new ProbabilisticAssociationVectorValidator(5));

        NetworkManager manager2 = new UnsupervisedRBMManager(rbm, database, 0, null,
                new ProbabilisticAssociationVectorValidator(5));

        manager.supervisedTraining(200_000);
        manager.testNetwork();
        manager2.supervisedTraining(200_000);
        manager2.testNetwork();
        LOGGER.info("Results: {}", manager.getTestResults().getComparableSuccess());
        LOGGER.info("Results: {}", manager2.getTestResults().getComparableSuccess());
    }

    public static void showMePowerOfDBN() {
        System.out.println("DBN power");
        long seed = 101;
        double weightsScale = 0.0;
        MnistDatasetReader reader = getReader(FULL_MNIST_SIZE, true);
        Database database = new Database(seed, reader.getTrainingData(), reader.getTestingData(), "MNIST", true);

        List<RestrictedBoltzmannMachine> rbms = new ArrayList<>();

        HeuristicDBN heuristic = new HeuristicDBN();
        heuristic.learningRate = 0.5;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.temperature = 1;
        heuristic.batchSize = 100;

        rbms.add(new BinaryRestrictedBoltzmannMachine("1. level", database.getSizeOfVector(), 100,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));
        /*rbms.add(new BinaryRestrictedBoltzmannMachine("2. level", 100, 100,
                new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed));*/
        StuckedRBM inputTransfer = new StuckedRBM(rbms, "NetworkChimney");
        StuckedRBMTrainer.train(inputTransfer, database, 100_000/*, 100_000*/);
        BinaryDeepBeliefNetwork theBeast = new BinaryDeepBeliefNetwork("theBeast", database.getNumberOfClasses(),
                500, new GaussianRndWeightsFactory(weightsScale, seed), heuristic, seed, inputTransfer);
        NetworkManager manager = new BinaryDeepBeliefNetworkManager(theBeast, database,
                null);
        MasterNetworkManager master = new MasterNetworkManager("dbn testing",
                Collections.singletonList(manager));

        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(10, 10000, master);
        benchmarker.benchmark();
    }

}
