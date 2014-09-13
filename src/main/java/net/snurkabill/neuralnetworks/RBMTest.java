package net.snurkabill.neuralnetworks;

import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.Tuple;
import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.TupleGenerator;
import net.snurkabill.neuralnetworks.data.DataItem;
import net.snurkabill.neuralnetworks.data.Database;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRBMManager;
import net.snurkabill.neuralnetworks.energybasednetwork.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.energybasednetwork.HeuristicParamsRBM;
import net.snurkabill.neuralnetworks.energybasednetwork.randomize.randomindexer.LinearRandomIndexer;
import net.snurkabill.neuralnetworks.feedforwardnetwork.weightfactory.GaussianRndWeightsFactory;
import net.snurkabill.neuralnetworks.results.UnsupervisedTestResults;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;

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
			
			for (int j = 0; j < asdf.size(); j++) {
                double[] trainValues = asdf.get(j).getDoubleArray();
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
        numOfHidden = 100;

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

        Map<Integer, List<?>> trainingSet = new HashMap<>();
        List<DataItem> first = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            first.add(new DataItem(firstVector));
        }
        List<DataItem> second = new ArrayList<>();
        for (int i = 0; i < 200; i++) {
            second.add(new DataItem(secondVector));
        }
        List<DataItem> third = new ArrayList<>();
        for (int i = 0; i < 300; i++) {
            third.add(new DataItem(thirdVector));
        }
        trainingSet.put(0, first);
        trainingSet.put(1, second);
        trainingSet.put(2, third);
        Database database = new Database(0, trainingSet, trainingSet, "testingDatabaseRBM");

        BinaryRBMManager manager = new BinaryRBMManager(Collections.singletonList(secondRbm), database, 0);

        for (int i = 0; i < 100; i++) {
            manager.trainNetworks(50);
            List<UnsupervisedTestResults> results = manager.testNetworks(0,
                    new LinearRandomIndexer(secondRbm.getSizeOfVisibleVector(), 0, 0));
            LOGGER.info("Success: {}", results.get(0).getSuccess());
        }
    }	
}
