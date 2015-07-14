package net.snurkabill.neuralnetworks.managers.concurrent;

import net.snurkabill.neuralnetworks.managers.NetworkManager;

import java.util.concurrent.Callable;

public class ConcurrentTrainer implements Callable<ConcurrentTrainer> {

    private final NetworkManager manager;
    private final int numberOfIterations;

    public ConcurrentTrainer(NetworkManager manager, int numberOfIterations) {
        this.manager = manager;
        this.numberOfIterations = numberOfIterations;
    }

    @Override
    public ConcurrentTrainer call() throws Exception {
        manager.trainNetwork(numberOfIterations);
        return this;
    }
}
