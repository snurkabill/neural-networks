package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.examples.gaussianvisible.GaussianVisibleImplementation;
import net.snurkabill.neuralnetworks.examples.mnist.DeepBeliefNetworkConcept;
import net.snurkabill.neuralnetworks.examples.mnist.MnistExampleFFNN;
import net.snurkabill.neuralnetworks.visualize.boltzmann.VisualizedBinaryRBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class App {

    private static final Logger LOGGER = LoggerFactory.getLogger("main");

    public static void main(String argv[]) throws IOException {
        //GaussianVisibleImplementation.shitFuck();
        //VisualizedBinaryRBM.start();
        DeepBeliefNetworkConcept.deepBoltzmannMachine();
        //DeepBeliefNetworkConcept.showMePowerOfDBN();
        //MnistExampleFFNN.basicBenchmark();
        //GaussianVisibleImplementation.shitFuck();
        //IWannaKillMyself.and();
    }
}
