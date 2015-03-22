package net.snurkabill.neuralnetworks.managers.concurrent;

import net.snurkabill.neuralnetworks.managers.NetworkManager;

public class ConcurrentTrainer implements Runnable {

    private final NetworkManager manager;
    private final int numberOfIterations;

    public ConcurrentTrainer(NetworkManager manager, int numberOfIterations) {
        this.manager = manager;
        this.numberOfIterations = numberOfIterations;
    }

    @Override
    public void run() {
        manager.supervisedTraining(numberOfIterations);
    }
}