package net.snurkabill.neuralnetworks.managers.concurrent;

import net.snurkabill.neuralnetworks.managers.NetworkManager;

import java.util.concurrent.Callable;

public class ConcurrentTester implements Callable<ConcurrentTester> {

    private final NetworkManager manager;

    public ConcurrentTester(NetworkManager manager) {
        this.manager = manager;
    }


    @Override
    public ConcurrentTester call() throws Exception {
        manager.testNetwork();
        return this;
    }
}
