package net.snurkabill.neuralnetworks;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRestrictedBolzmannMachine;
import net.snurkabill.neuralnetworks.energybasednetwork.HeuristicParamsRBM;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class RBMTest {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("RBM test");
	
	public void test() {

		int numOfVisible = 2;
		int numOfHidden = 10;

		BinaryRestrictedBolzmannMachine rbm = new BinaryRestrictedBolzmannMachine(numOfVisible, numOfHidden, 
				new GaussianRndWeightsFactory(0.001, 0), HeuristicParamsRBM.createBasicHeuristicParams(), 0);
		
		List<Tuple> asdf = new ArrayList<>();
		
		for (int i = 0; i < 4; i++) {
			asdf.add(new Tuple(1,1));
		}
		
		for (int i = 0; i < 3; i++) {
			asdf.add(new Tuple(1,0));
		}
		
		for (int i = 0; i < 2; i++) {
			asdf.add(new Tuple(0,1));
		}
		
		for (int i = 0; i < 1; i++) {
			asdf.add(new Tuple(0,0));
		}
		
		
		int numOfIterations = 100;
		for (int i = 0; i < numOfIterations; i++) {
			Collections.shuffle(asdf);
			
			double[] trainValues = new double[rbm.getSizeOfVisibleVector()];
			for (int j = 0; j < asdf.size(); j++) {
				trainValues[0] = asdf.get(j).a;
				trainValues[1] = asdf.get(j).b;
				rbm.trainMachine(trainValues);
				LOGGER.info("Energy = {}", rbm.calcEnergy());
			}
		}
		
		int[] counter = new int[4];
		
		int numOfTestingIterations = 100000;
		for (int i = 0; i < numOfTestingIterations; i++) {
			
			LOGGER.info("{}", rbm.reconstructNext());
			//Tuple tuple = new Tuple(rbm.reconstructNext());
			
			
			//counter[tuple.num]++;
		}
		LOGGER.info("RESULTS: {}", counter);
		
	}	
}
