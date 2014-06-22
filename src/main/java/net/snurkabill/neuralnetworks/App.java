package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.examples.mnist.MnistExampleFFNN;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class App {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("main");
	
	public static void main (String argv[]) {
		
		new RBMTest().test();
		
		MnistExampleFFNN.startExample();
	}
}
