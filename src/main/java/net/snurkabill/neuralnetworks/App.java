package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.examples.mnist.BinaryMnistRBM;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.IOException;

public class App {

    private static final Logger LOGGER = LoggerFactory.getLogger("main");

    public static void main(String argv[]) throws IOException {

        //GaussianVisibleTest.gaussianVisibleExample();


        //MnistExampleFFNN.mnistDBN();

        BinaryMnistRBM.start();
        //new RBMTest().test();

        //DeepBeliefNetworkConcept.errorFunctionOnRBMs();
        //DeepBeliefNetworkConcept.deepXor();
        //DeepBeliefNetworkConcept.deepBoltzmannMachine();

//        MnistExampleFFNN.startExample();
//        MnistExampleFFNN.basicBenchmark();
        //MnistExampleFFNN.useRelu();
        //LinearVisibleTest.test();
    }
}
