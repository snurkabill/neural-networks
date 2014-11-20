package net.snurkabill.neuralnetworks.managers.deep;

import net.snurkabill.neuralnetworks.data.database.DataItem;
import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.neuralnetwork.deep.StuckedRBM;

import java.util.Iterator;

public class StuckedRBMTrainer {

    public static void train(StuckedRBM networkChimney, Database database, int... trainingIterationsPerLevel) {
        if(trainingIterationsPerLevel.length != networkChimney.getNumOfLevels()) {
            throw new IllegalArgumentException("Num of RBms is different than array with training iterations");
        }

        Iterator<DataItem> trainingIterator = database.getInfiniteTrainingIterator();
        for (int i = 0; i < trainingIterationsPerLevel.length; i++) {
            for (int j = 0; j < trainingIterationsPerLevel[i]; j++) {
                networkChimney.trainNetworkLevel(trainingIterator.next().data, i);
            }
        }
    }
}
