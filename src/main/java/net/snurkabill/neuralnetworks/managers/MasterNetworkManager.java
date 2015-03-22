package net.snurkabill.neuralnetworks.managers;

import net.snurkabill.neuralnetworks.managers.concurrent.ConcurrentTester;
import net.snurkabill.neuralnetworks.managers.concurrent.ConcurrentTrainer;
import net.snurkabill.neuralnetworks.results.ResultsSummary;
import net.snurkabill.neuralnetworks.utilities.Timer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class MasterNetworkManager {

    private final Logger LOGGER;
    private final List<NetworkManager> managers;
    private final Timer timer = new Timer();
    private final String nameOfProblem;
    private final ExecutorService executor;
    private final List<Future> futureManagers;

    public MasterNetworkManager(String nameOfProblem, List<NetworkManager> singleNetworkManagers, int numberOfThreads) {
        this.LOGGER = LoggerFactory.getLogger(MasterNetworkManager.class.getSimpleName() + ": " + nameOfProblem);
        this.managers = singleNetworkManagers;
        this.nameOfProblem = nameOfProblem;
        this.executor = Executors.newFixedThreadPool(numberOfThreads);
        this.futureManagers = new ArrayList<>(singleNetworkManagers.size());
    }

    public String getNameOfProblem() {
        return nameOfProblem;
    }

    public int getNumOfNetworks() {
        return managers.size();
    }

    public void trainNetworks(int numOfIterations) {
        LOGGER.info("================================================================================");
        LOGGER.info("TRAINING STARTED");
        timer.startTimer();
        for (int i = 0; i < futureManagers.size(); i++) {
            futureManagers.set(i, executor.submit(new ConcurrentTrainer(managers.get(i), numOfIterations)));
        }
        executor.shutdown();
        for (Future future : futureManagers) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                LOGGER.error("The " + this.getClass().getSimpleName() + " of problem: " + this.getNameOfProblem() + " "
                        + "was interrupted during training phase.", e);
                throw new RuntimeException(e);
            }
        }
        timer.stopTimer();
        LOGGER.info("TRAINING STOPPED; TOTAL TIME: {} seconds", timer.secondsSpent());
        LOGGER.info("================================================================================");
    }

    public void testNetworks() {
        LOGGER.info("================================================================================");
        LOGGER.info("TESTING STARTED");
        timer.startTimer();
        for (int i = 0; i < futureManagers.size(); i++) {
            futureManagers.set(i, executor.submit(new ConcurrentTester(managers.get(i))));
        }
        executor.shutdown();
        for (Future future : futureManagers) {
            try {
                future.get();
            } catch (InterruptedException | ExecutionException e) {
                LOGGER.error("The " + this.getClass().getSimpleName() + " of problem: " + this.getNameOfProblem() + " "
                        + "was interrupted during testing phase.", e);
                throw new RuntimeException(e);
            }
        }
        timer.stopTimer();
        LOGGER.info("TESTING STOPPED; TOTAL TIME: {} seconds", timer.secondsSpent());
        LOGGER.info("================================================================================");
    }

    public List<ResultsSummary> getResults() {
        List<ResultsSummary> results = new ArrayList<>(managers.size());
        for (NetworkManager manager : managers) {
            results.add(manager.getResults());
        }
        return results;
    }
}
