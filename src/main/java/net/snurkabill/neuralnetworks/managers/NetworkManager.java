package net.snurkabill.neuralnetworks.managers;


import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.DeepBoltzmannMachine;
import net.snurkabill.neuralnetworks.results.NetworkResults;
import net.snurkabill.neuralnetworks.results.ResultsSummary;
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
    protected final Database.InfiniteRandomTrainingIterator infiniteTrainingIterator;
    protected final List<NetworkResults> results = new ArrayList<>();
    private final Timer timer = new Timer();
    protected double globalError;
    protected double percentageSuccess;
    protected int learnedVectorsBeforeTest;
    protected long learningTimeBeforeTest;
    private final HeuristicCalculator heuristicCalculator;
    private boolean wasAlreadyTested;
    protected final int[][] confusionMatrix;
    protected final double[][] confusionPercentages;

    public NetworkManager(NeuralNetwork neuralNetwork, Database database,
                          HeuristicCalculator heuristicCalculator) {
        this.neuralNetwork = neuralNetwork;
        this.LOGGER = LoggerFactory.getLogger(neuralNetwork.getName());
        this.database = database;
        this.heuristicCalculator = heuristicCalculator;
        checkVectorSizes();
        // TODO: general EVALUATOR!
        this.targetMaker = new SeparableTargetValues(neuralNetwork.getTransferFunction(),
                database.getNumberOfClasses());
        this.infiniteTrainingIterator = database.getInfiniteRandomTrainingIterator();
        this.confusionMatrix = new int[database.getNumberOfClasses()][database.getNumberOfClasses()];
        this.confusionPercentages = new double[database.getNumberOfClasses()][database.getNumberOfClasses()];
    }

    public void supervisedTraining(int numOfIterations) {
        if ((neuralNetwork instanceof DeepBoltzmannMachine)) {
            LOGGER.warn("Neural network {} is already trained and can't be trained anymore, skipping");
            return;
        }
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
        if ((neuralNetwork instanceof DeepBoltzmannMachine) && wasAlreadyTested) {
            this.results.add(results.get(0)); // hardly coded! .... I don't like this
        }
        timer.startTimer();
        test();
        timer.stopTimer();
        processResults();
        this.learnedVectorsBeforeTest = 0;
        this.learningTimeBeforeTest = 0;
        this.wasAlreadyTested = true;
        LOGGER.info("Testing {} samples took {} seconds, {} samples/sec",
                database.getTestSetSize(), timer.secondsSpent(), timer.samplesPerSec(database.getTestSetSize()));
        if (heuristicCalculator != null) {
            neuralNetwork.setHeuristic(heuristicCalculator.calculateNewHeuristic(results));
        }
        LOGGER.info("Global Error: {}, Percentage Successs: {}", globalError, percentageSuccess);

        if(LOGGER.isDebugEnabled()) {
            for (int i = 0; i < database.getNumberOfClasses(); i++) {
                for (int j = 0; j < database.getNumberOfClasses(); j++) {
                    confusionPercentages[i][j] = (confusionMatrix[i][j] * 100.0) / database.getSizeOfTestingDataset(i);
                }
            }
            LOGGER.debug("Confusion Matrix:");
            for (int i = 0; i < database.getNumberOfClasses(); i++) {
                LOGGER.debug("{}: {}, {}", i, confusionMatrix[i], confusionPercentages[i]);
            }
        }
    }

    protected abstract void train(int numOfIterations);

    protected abstract void test();

    protected abstract void processResults();

    protected abstract void checkVectorSizes();

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
