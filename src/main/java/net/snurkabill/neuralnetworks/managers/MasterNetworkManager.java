package net.snurkabill.neuralnetworks.managers;

import net.snurkabill.neuralnetworks.results.ResultsSummary;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;

public class MasterNetworkManager {

    private final Logger LOGGER;
    private final List<NetworkManager> managers;
    private final Timer timer = new Timer();
    private final String nameOfProblem;

    public MasterNetworkManager(String nameOfProblem, List<NetworkManager> singleNetworkManagers) {
        this.LOGGER = LoggerFactory.getLogger(MasterNetworkManager.class.getSimpleName() + ": " + nameOfProblem);
        this.managers = singleNetworkManagers;
        this.nameOfProblem = nameOfProblem;
    }

    public String getNameOfProblem() {
        return nameOfProblem;
    }

    public int getNumOfNetworks() {
        return managers.size();
    }

    public void trainNetworks(int numOfIterations) {
        LOGGER.info("------------------------------TRAINING-STARTED------------------------------");
        timer.startTimer();
        for (NetworkManager manager : managers) {
            manager.supervisedTraining(numOfIterations);
        }
        timer.stopTimer();
        LOGGER.info("TOTAL TIME: {} seconds", timer.secondsSpent());
        LOGGER.info("------------------------------TRAINING-STOPPED------------------------------");
    }

    public void testNetworks() {
        LOGGER.info("------------------------------TESTING-STARTED-------------------------------");
        timer.startTimer();
        for (NetworkManager manager : managers) {
            manager.testNetwork();
        }
        timer.stopTimer();
        LOGGER.info("TOTAL TIME: {} seconds", timer.secondsSpent());
        LOGGER.info("------------------------------TESTING-STOPPED-------------------------------");
    }

    public List<ResultsSummary> getResults() {
        List<ResultsSummary> results = new ArrayList<>(managers.size());
        for (NetworkManager manager : managers) {
            results.add(manager.getResults());
        }
        return results;
    }
}
