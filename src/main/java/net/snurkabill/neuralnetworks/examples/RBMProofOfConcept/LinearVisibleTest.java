package net.snurkabill.neuralnetworks.examples.RBMProofOfConcept;

import net.snurkabill.neuralnetworks.benchmark.SupervisedBenchmarker;
import net.snurkabill.neuralnetworks.data.mnist.MnistDatasetReader;
import net.snurkabill.neuralnetworks.examples.mnist.MnistExampleFFNN;
import net.snurkabill.neuralnetworks.heuristic.BoltzmannMachineHeuristic;
import net.snurkabill.neuralnetworks.managers.MasterNetworkManager;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.managers.supervised.classification.RBMClassificationNetworkManager;
import net.snurkabill.neuralnetworks.math.function.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.impl.linearvisible.LinearToLinearRBM;
import net.snurkabill.neuralnetworks.newdata.database.ClassificationDatabase;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.weights.weightfactory.GaussianRndWeightsFactory;

import java.util.Arrays;

public class LinearVisibleTest {

    public static void test() {
        long seed = 0;
        MnistDatasetReader reader = MnistExampleFFNN.getReader(60_000, false);

        TransferFunctionCalculator calculator2 = new SigmoidFunction();
        NewDatabase database2 = new ClassificationDatabase(reader.getTrainingData(), reader.getValidationData(),
                reader.getTestingData(), calculator2, "MNIST", seed);
        BoltzmannMachineHeuristic heuristic = BoltzmannMachineHeuristic.createStartingHeuristicParams();
        heuristic.learningRate = 0.000000001;
        heuristic.constructiveDivergenceIndex = 1;
        heuristic.batchSize = 30;
        heuristic.momentum = 0.1;
        heuristic.temperature = 1;
        heuristic.l2RegularizationConstant = 0.999;
        heuristic.dynamicKillingWeights = true;
        RestrictedBoltzmannMachine machine =
                new LinearToLinearRBM("RBM 1",
                        (database2.getVectorSize() + database2.getTargetSize()), 200,
                        new GaussianRndWeightsFactory(0.0001, seed),
                        heuristic, seed);
        NetworkManager manager_rbm = new RBMClassificationNetworkManager(machine, database2, null,
                NetworkManager.TrainingMode.STOCHASTIC);

        MasterNetworkManager superManager = new MasterNetworkManager("MNIST",
                Arrays.asList(manager_rbm), 1);
        SupervisedBenchmarker benchmarker = new SupervisedBenchmarker(5, 1000, superManager);
        benchmarker.benchmark();

    }
}
