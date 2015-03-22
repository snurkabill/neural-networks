package net.snurkabill.neuralnetworks.managers.concurrent;

import net.snurkabill.neuralnetworks.managers.NetworkManager;

public class ConcurrentTester implements Runnable {

    private final NetworkManager manager;

    public ConcurrentTester(NetworkManager manager) {
        this.manager = manager;
    }

    @Override
    public void run() {
        manager.testNetwork();
    }
}
