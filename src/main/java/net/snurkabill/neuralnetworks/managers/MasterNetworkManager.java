package net.snurkabill.neuralnetworks.managers;

import net.snurkabill.neuralnetworks.managers.concurrent.ConcurrentValidator;
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
    private ExecutorService executor;
    private final int numberOfThreads;

    public MasterNetworkManager(String nameOfProblem, List<NetworkManager> singleNetworkManagers, int numberOfThreads) {
        this.LOGGER = LoggerFactory.getLogger(MasterNetworkManager.class.getSimpleName() + ": " + nameOfProblem);
        this.managers = singleNetworkManagers;
        this.nameOfProblem = nameOfProblem;
        this.numberOfThreads = numberOfThreads > Runtime.getRuntime().availableProcessors() ?
                Runtime.getRuntime().availableProcessors() : numberOfThreads;
        LOGGER.info("Master manager created with {} threads", this.numberOfThreads);
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
        this.executor = Executors.newFixedThreadPool(numberOfThreads);
        List<Future<ConcurrentTrainer>> tasks = new ArrayList<>();
        for (NetworkManager manager : managers) {
            tasks.add(executor.submit(new ConcurrentTrainer(manager, numOfIterations)));
        }
        executor.shutdown();
        for (Future<ConcurrentTrainer> task : tasks) {
            try {
                task.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Training phase exception", e);
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
        this.executor = Executors.newFixedThreadPool(numberOfThreads);
        List<Future<ConcurrentValidator>> tasks = new ArrayList<>();
        for (NetworkManager manager : managers) {
            tasks.add(executor.submit(new ConcurrentValidator(manager)));
        }
        executor.shutdown();
        for (Future<ConcurrentValidator> task : tasks) {
            try {
                task.get();
            } catch (InterruptedException | ExecutionException e) {
                throw new RuntimeException("Testing phase excetion", e);
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
