package net.snurkabill.neuralnetworks.managers.deep;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Iterator;

public class StuckedRBMTrainer {

    public static final Logger LOGGER = LoggerFactory.getLogger("StuckerRBM trainer");

    public static void train(StuckedRBM networkChimney, Database database, int... trainingIterationsPerLevel) {
        if (trainingIterationsPerLevel.length != networkChimney.getNumOfLevels()) {
            throw new IllegalArgumentException("Num of RBms is different than array with training iterations");
        }
        Iterator<DataItem> trainingIterator = database.getInfiniteTrainingIterator();
        LOGGER.info("Training {} RBMs have started!", networkChimney.getNumOfLevels());
        Timer timer = new Timer();
        long totalTime = 0;
        for (int i = 0; i < trainingIterationsPerLevel.length; i++) {
            timer.startTimer();
            for (int j = 0; j < trainingIterationsPerLevel[i]; j++) {
                networkChimney.trainNetworkLevel(trainingIterator.next().data, i);
            }
            timer.stopTimer();
            LOGGER.info("Training {} level on {} samples took {} sec, {} samples/sec, ", i,
                    trainingIterationsPerLevel[i], timer.secondsSpent(),
                    timer.samplesPerSec(trainingIterationsPerLevel[i]));
            totalTime += timer.secondsSpent();
        }
        LOGGER.info("Training stucked RMBs is done! Total seconds: {}", totalTime);
    }
}
