package net.snurkabill.neuralnetworks.managers.concurrent;

import net.snurkabill.neuralnetworks.managers.NetworkManager;

import java.util.concurrent.Callable;

public class ConcurrentValidator implements Callable<ConcurrentValidator> {

    private final NetworkManager manager;

    public ConcurrentValidator(NetworkManager manager) {
        this.manager = manager;
    }

    @Override
    public ConcurrentValidator call() throws Exception {
        manager.validateNetwork();
        return this;
    }
}
