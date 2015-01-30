package net.snurkabill.neuralnetworks.managers.deep;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.DeepNetUtils;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;

public class GreedyLayerWiseStackedDeepRBMTrainer {

    public static final Logger LOGGER = LoggerFactory.getLogger("GreedyLayerWiseStackedDeepTrainer");

    public static List<RestrictedBoltzmannMachine> train(List<RestrictedBoltzmannMachine> machines,
                                                         List<Integer> iterationsForLevels, Database database) {
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
        for (int i = 0; i < machines.size(); i++) {
            LOGGER.info("Training {}. layer", i);
            Timer layerTimer = new Timer();
            layerTimer.startTimer();
            for (int j = 0; j < iterationsForLevels.get(i); j++) {
                machines.get(0).setVisibleNeurons(database.getRandomizedTrainingData().data);
                for (int k = 0; k < i; k++) {
                    machines.get(k + 1).setVisibleNeurons(machines.get(k).activateHiddenNeurons());
                }
                machines.get(i).trainMachine();
            }
            layerTimer.stopTimer();
            LOGGER.info("Training {}. level on {} samples took {} sec, {} samples/sec, ", i, iterationsForLevels.get(i),
                    layerTimer.secondsSpent(), layerTimer.samplesPerSec(iterationsForLevels.get(i)));
        }
        totalTime.stopTimer();
        LOGGER.info("Training finished. Total time: {}, trained iterations per levels: {}", totalTime.secondsSpent(),
                iterationsForLevels);
        return machines;
    }
}
