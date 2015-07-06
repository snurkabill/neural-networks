package net.snurkabill.neuralnetworks.managers.deep;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.RBMCone;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.DeepNetUtils;
import net.snurkabill.neuralnetworks.utilities.Timer;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class GreedyLayerWiseDeepBeliefNetworkBuilder {

    public static final Logger LOGGER = LoggerFactory.getLogger("GreedyLayerWiseStackedDeepTrainer");

    public class NormalizerVector {
        private final double[] stdDev;
        private final double[] mean;

        public NormalizerVector(double[] stdDev, double[] mean) {
            this.stdDev = stdDev;
            this.mean = mean;
        }

        public double[] getStdDev() {
            return stdDev;
        }

        public double[] getMean() {
            return mean;
        }

        public double[] normalize(double[] vector) {
            double[] normalizedVector = new double[vector.length];
            for (int i = 0; i < vector.length; i++) {
                if (stdDev[i] == 0) {
                    normalizedVector[i] = 0.0;
                } else {
                    normalizedVector[i] = (vector[i] - mean[i]) / stdDev[i];
                }
            }
            return normalizedVector;
        }
    }

    private final List<RestrictedBoltzmannMachine> machines;
    private final List<Integer> iterationsForLevels;
    private final Database database;
    private final List<NormalizerVector> normalizerLayers = new ArrayList<>();
    private final double percentageOfDatabaseForNormalization;

    public GreedyLayerWiseDeepBeliefNetworkBuilder(List<RestrictedBoltzmannMachine> machines,
                                                   List<Integer> iterationsForLevels,
                                                   double percentageOfDatabaseForNormalization,
                                                   Database database) {
        this.machines = machines;
        this.iterationsForLevels = iterationsForLevels;
        this.database = database;
        this.percentageOfDatabaseForNormalization = percentageOfDatabaseForNormalization;
    }

    public RBMCone createRBMCone() {
        DeepNetUtils.checkMachineList(machines);
        if (iterationsForLevels == null) {
            throw new IllegalArgumentException("List of machines is null");
        }
        if (iterationsForLevels.isEmpty()) {
            throw new IllegalArgumentException("List of iterations is empty");
        }
        if (machines.size() != iterationsForLevels.size()) {
            throw new IllegalStateException("Difference between machines size and iterations size");
        }
        if (database.getSizeOfVector() != machines.get(0).getVisibleNeurons().length) {
            throw new IllegalStateException("First RBM has different size of input vector than database has");
        }
        LOGGER.info("Training {} RBMs started. Total iterations: {}", machines.size(), iterationsForLevels);
        Timer totalTime = new Timer();
        totalTime.startTimer();
        Database.InfiniteRandomTrainingIterator iterator = database.getInfiniteRandomTrainingIterator();
        for (int i = 0; i < machines.size(); i++) {
            createNormalizer(iterator, i);
            trainLayer(iterator, i);
        }
        totalTime.stopTimer();
        LOGGER.info("Training finished. Total time: {}, trained iterations per levels: {}", totalTime.secondsSpent(),
                iterationsForLevels);

        return new RBMCone(machines, normalizerLayers);
    }

    private void feedForward(double[] vector, int stoppingLayerIndex) {
        machines.get(0).setVisibleNeurons(normalizerLayers.get(0).normalize(vector));
        for (int k = 0; k < stoppingLayerIndex; k++) {
            machines.get(k + 1).setVisibleNeurons(normalizerLayers.get(k + 1).normalize(machines.get(k).activateHiddenNeurons()));
        }
    }

    private double[] feedForwardAndActivateLastHidden(double[] vector, int stoppingLayerIndex) {
        if(stoppingLayerIndex == 0) {
            return vector;
        }
        feedForward(vector, stoppingLayerIndex - 1);
        return machines.get(stoppingLayerIndex - 1).activateHiddenNeurons();
    }

    private int resolveNumOfSampling() {
        return (int) (database.getTrainingSetSize() * percentageOfDatabaseForNormalization >
                database.getNumberOfClasses() ?
                database.getTrainingSetSize() * percentageOfDatabaseForNormalization :
                database.getNumberOfClasses());
    }

    private void createNormalizer(Iterator<DataItem> randomIterator, int layerIndex) {
        LOGGER.info("Calculating normalization vector of {}th level", layerIndex);
        Timer timer = new Timer();
        timer.startTimer();
        double[] stdev = new double[machines.get(layerIndex).getSizeOfVisibleVector()];
        double[] mean = new double[machines.get(layerIndex).getSizeOfVisibleVector()];
        for (int i = 0; i < machines.get(layerIndex).getSizeOfVisibleVector(); i++) {
            int sampleSize = resolveNumOfSampling();
            double[] array = new double[sampleSize];
            for (int j = 0; j < sampleSize; j++) {
                array[j] = feedForwardAndActivateLastHidden(randomIterator.next().data, layerIndex)[i];
            }
            mean[i] = Utilities.mean(array);
            stdev[i] = Utilities.stddev(array, mean[i]);
        }
        normalizerLayers.add(new NormalizerVector(stdev, mean));
        timer.stopTimer();
        LOGGER.info("Normalization stopped after {} seconds", timer.secondsSpent());
    }

    private void trainLayer(Iterator<DataItem> iterator, int layerIndex) {
        LOGGER.info("Training {}th layer", layerIndex);
        Timer layerTimer = new Timer();
        layerTimer.startTimer();
        for (int iteration = 0; iteration < iterationsForLevels.get(layerIndex); iteration++) {
            feedForward(iterator.next().data, layerIndex);
            machines.get(layerIndex).trainMachine();
        }
        layerTimer.stopTimer();
        LOGGER.info("Training {}. level on {} samples took {} sec, {} samples/sec, ", layerIndex,
                iterationsForLevels.get(layerIndex), layerTimer.secondsSpent(),
                layerTimer.samplesPerSec(iterationsForLevels.get(layerIndex)));
    }

}
