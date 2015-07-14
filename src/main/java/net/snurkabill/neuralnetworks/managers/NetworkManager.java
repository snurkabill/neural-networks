package net.snurkabill.neuralnetworks.managers;


import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import net.snurkabill.neuralnetworks.newdata.database.NewDatabase;
import net.snurkabill.neuralnetworks.results.NetworkResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Enumeration;
import java.util.List;

public abstract class NetworkManager {

    public enum TrainingMode {
        STOCHASTIC,
        SEQUENTIAL
    }

    public enum TestingMode {
        VALIDATION,
        TESTING
    }

    protected final Logger LOGGER;
    protected final NeuralNetwork neuralNetwork;
    protected final NewDatabase database;
    protected final Enumeration<NewDataItem> dataEnumerator;
    protected final List<NetworkResults> results = new ArrayList<>();
    private final Timer timer = new Timer();
    protected double globalError;
    protected int learnedVectorsBeforeTest;
    protected long learningTimeBeforeTest;
    private final HeuristicCalculator heuristicCalculator;
    protected final TrainingMode trainingMode;

    public NetworkManager(NeuralNetwork neuralNetwork, NewDatabase database,
                          HeuristicCalculator heuristicCalculator, TrainingMode trainingMode) {
        this.neuralNetwork = neuralNetwork;
        this.LOGGER = LoggerFactory.getLogger(neuralNetwork.getName());
        this.database = database;
        this.heuristicCalculator = heuristicCalculator;
        this.trainingMode = trainingMode;
        if(trainingMode == TrainingMode.STOCHASTIC) {
            this.dataEnumerator = database.createInfiniteRandomTrainingSetEnumerator();
        } else if(trainingMode == TrainingMode.SEQUENTIAL) {
            this.dataEnumerator = database.createInfiniteSimpleTrainingSetEnumerator();
        } else {
            throw new IllegalArgumentException("There is no other option yet");
        }
    }

    public void trainNetwork(int numOfIterations) {
        if (numOfIterations < database.getTargetSize()) {
            LOGGER.warn("Count of iterations for training([{}]) is smaller than size of targetVector([{}]). " +
                            "Ineffective training possible: setting numOfIterations to targetVector", numOfIterations,
                    database.getTargetSize());
            numOfIterations = database.getTargetSize();
        }
        if (numOfIterations % database.getTargetSize() != 0) {
            LOGGER.warn("Count of iterations for training([{}]) is not dividable by size of targetVector ([{}]). " +
                    "Rounding up!", numOfIterations, database.getTargetSize());
            numOfIterations = (numOfIterations / database.getTargetSize()) + database.getTargetSize();
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

    public void validateNetwork() {
        analyzeNetwork(TestingMode.VALIDATION);
    }

    public void testNetwork() {
        analyzeNetwork(TestingMode.TESTING);
    }

    private void analyzeNetwork(TestingMode testingMode) {
        timer.startTimer();
        test(testingMode);
        timer.stopTimer();
        processResults();
        this.learnedVectorsBeforeTest = 0;
        this.learningTimeBeforeTest = 0;
        LOGGER.info("Testing {} samples took {} seconds, {} samples/sec",
                database.getTestingSetSize(), timer.secondsSpent(), timer.samplesPerSec(database.getTestingSetSize()));
        if (heuristicCalculator != null) {
            LOGGER.info("Recalculating heuristic");
            neuralNetwork.setHeuristic(heuristicCalculator.calculateNewHeuristic(results));
        }
        LOGGER.info("Global Error: {}", globalError);
        printMoreInfo();
    }

    protected abstract void train(int numOfIterations);

    protected abstract void test(TestingMode testingMode);

    protected abstract void processResults();

    protected abstract void printMoreInfo();

    public NetworkResults getTestResults() {
        return this.results.get(this.results.size() - 1);
    }

    public ResultsSummary getResults() {
        ResultsSummary results = new ResultsSummary(neuralNetwork.determineWorkingName());
        for (NetworkResults results_ : this.results) {
            results.add(results_);
        }
        return results;
    }
}
