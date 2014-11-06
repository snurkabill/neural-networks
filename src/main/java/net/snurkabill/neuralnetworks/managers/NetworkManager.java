package net.snurkabill.neuralnetworks.managers;


import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.BinaryRestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.FeedForwardableNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.feedforward.transferfunction.SigmoidFunction;
import net.snurkabill.neuralnetworks.results.ResultsSummary;
import net.snurkabill.neuralnetworks.results.TestResults;
import net.snurkabill.neuralnetworks.target.SeparableTargetValues;
import net.snurkabill.neuralnetworks.target.TargetValues;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public abstract class NetworkManager {

    protected final Logger LOGGER;
    protected final NeuralNetwork neuralNetwork;
    protected final Database database;
    protected final TargetValues targetMaker;
    protected final Database.InfiniteSimpleTrainSetIterator infiniteTrainingIterator;
    protected final List<TestResults> results = new ArrayList<>();
    private final Timer timer = new Timer();
    protected double globalError;
    protected double percentageSuccess;
    protected int learnedVectorsBeforeTest;
    protected long learningTimeBeforeTest;
    private final HeuristicCalculator heuristicCalculator;

    public NetworkManager(NeuralNetwork neuralNetwork, Database database,
                          HeuristicCalculator heuristicCalculator) {
        this.neuralNetwork = neuralNetwork;
        this.LOGGER = LoggerFactory.getLogger(neuralNetwork.getName());
        this.database = database;
        this.heuristicCalculator = heuristicCalculator;
        // TODO: general EVALUATOR!
        if (neuralNetwork instanceof BinaryRestrictedBoltzmannMachine) {
            this.targetMaker = new SeparableTargetValues(new SigmoidFunction(), database.getNumberOfClasses());
        } else {
            this.targetMaker = new SeparableTargetValues(((FeedForwardableNetwork) neuralNetwork).getTransferFunction(),
                    database.getNumberOfClasses());
        }
        this.infiniteTrainingIterator = database.getInfiniteTrainingIterator();
    }

    public void supervisedTraining(int numOfIterations) {
        if (numOfIterations < database.getNumberOfClasses()) {
            LOGGER.warn("Count of iterations for training([{}]) is smaller than number of classes of division([{}]). " +
                            "Uneffective training possible: setting numOfIterations to numberOfClasses", numOfIterations,
                    database.getNumberOfClasses());
            numOfIterations = database.getNumberOfClasses();
        }
        if (numOfIterations % database.getNumberOfClasses() != 0) {
            LOGGER.warn("Count of iterations for training([{}]) is not dividable by count of classes of " +
                    "division([{}]). Rounding up!", numOfIterations, database.getNumberOfClasses());
            numOfIterations = (numOfIterations / database.getNumberOfClasses()) + database.getNumberOfClasses();
            // integer division + numOfClasses -> rounding up!
        }
        timer.startTimer();
        train(numOfIterations);
        timer.stopTimer();
        learnedVectorsBeforeTest += numOfIterations;
        learningTimeBeforeTest += timer.getTotalTime();
        LOGGER.info("Training {} samples took {} seconds, {} samples/sec",
                numOfIterations, timer.secondsSpent(), timer.samplesPerSec(numOfIterations));
    }

    public void testNetwork() {
        timer.startTimer();
        test();
        timer.stopTimer();
        processResults();
        this.learnedVectorsBeforeTest = 0;
        this.learningTimeBeforeTest = 0;
        LOGGER.info("Testing {} samples took {} seconds, {} samples/sec",
                database.getTestSetSize(), timer.secondsSpent(), timer.samplesPerSec(database.getTestSetSize()));
        if(heuristicCalculator != null) {
            neuralNetwork.setHeuristic(heuristicCalculator.calculateNewHeuristic(results));
        }
    }

    protected abstract void train(int numOfIterations);

    protected abstract void test();

    protected abstract void processResults();

    public TestResults getTestResults() {
        return this.results.get(this.results.size() - 1);
    }

    public ResultsSummary getResults() {
        ResultsSummary results = new ResultsSummary(neuralNetwork.determineWorkingName());
        for (TestResults results_ : this.results) {
            results.add(results_);
        }
        return results;
    }
}
