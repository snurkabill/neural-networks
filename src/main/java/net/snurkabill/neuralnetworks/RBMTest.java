package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.Tuple;
import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.TupleGenerator;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.energybasednetwork.HeuristicParamsRBM;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class RBMTest {
	
	private static final Logger LOGGER = LoggerFactory.getLogger("RBM test");


	public void test() {

        LOGGER.info("Testing tuples generator");

        List<Tuple> tuples = TupleGenerator.createTuples(3);

        LOGGER.info("{}", tuples);

		int numOfVisible = 2;
		int numOfHidden = 10;

		BinaryRestrictedBoltzmannMachine rbm = new BinaryRestrictedBoltzmannMachine(numOfVisible, numOfHidden, 
				new GaussianRndWeightsFactory(0.001, 0), HeuristicParamsRBM.createBasicHeuristicParams(), 0);
		
		List<Tuple> asdf = new ArrayList<>();

        int[] tupleArray = {1, 0};

        Tuple ttuple = new Tuple(tupleArray);
        LOGGER.info("Tuple: {}, num: {}", tupleArray, ttuple.toString());

        int [] tupleArray2 = {1, 0, 0};
        ttuple = new Tuple(tupleArray2);

        LOGGER.info("Tuple: {}, num: {}", tupleArray2, ttuple.toString());


        int[] tupleArray3 = {1, 0, 0, 1};
        ttuple = new Tuple(tupleArray3);

        LOGGER.info("Tuple: {}, num: {}", tupleArray3, ttuple.toString());

		for (int i = 0; i < 4; i++) {
			asdf.add(new Tuple(1, 1));
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
		
		int numOfIterations = 10000;
		for (int i = 0; i < numOfIterations; i++) {
			Collections.shuffle(asdf);
			
			double[] trainValues = new double[rbm.getSizeOfVisibleVector()];
			for (int j = 0; j < asdf.size(); j++) {
				trainValues = asdf.get(j).getDoubleArray();
				rbm.trainMachine(trainValues);
				//LOGGER.info("Energy = {}", rbm.calcEnergy());
			}
		}
		
		int[] counter = new int[10];
		
		int numOfTestingIterations = 100000;
		for (int i = 0; i < numOfTestingIterations; i++) {
			
			//LOGGER.info("{}", rbm.reconstructNext());
			Tuple tuple = new Tuple(rbm.reconstructNext());
			counter[tuple.getNum()]++;
            //LOGGER.info("[{}, {}]", tuple.a, tuple.b);
		}
		LOGGER.info("RESULTS: {}", counter);

        LOGGER.info("NEXT TEST");

        numOfVisible = 20;
        numOfHidden = 20;

        BinaryRestrictedBoltzmannMachine secondRbm = new BinaryRestrictedBoltzmannMachine(numOfVisible, numOfHidden,
                new GaussianRndWeightsFactory(0.01, 0), HeuristicParamsRBM.createBasicHeuristicParams(), 0);

        double[] firstVector =
                {1, 1, 1, 1, 1,
                1, 1, 1, 1, 1,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0};

        double[] secondVector =
                {1, 0, 1, 0, 1,
                0, 1, 0, 1, 0,
                1, 0, 1, 0, 1,
                0, 1, 0, 1, 0};

        double[] thirdVector =
                {0, 1, 0, 1, 0,
                1, 0, 1, 0, 1,
                0, 1, 0, 1, 0,
                1, 0, 1, 0, 1};

        List<double[]> asdfasdf = new ArrayList<>();
        asdfasdf.add(firstVector);
        asdfasdf.add(secondVector);
        asdfasdf.add(thirdVector);

        numOfIterations = 100;

        double[] last = null;
        for (int i = 0; i < numOfIterations; i++) {
            Collections.shuffle(asdfasdf);

            for (double[] kkk : asdfasdf) {
                secondRbm.trainMachine(kkk);
                last = kkk;
            }
        }

        for (int i = 0; i < 100; i++) {
            LOGGER.info("{}", secondRbm.reconstructNext());
        }
        LOGGER.info("LAST: {}", last);
    }	
}
