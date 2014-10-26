package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;

public abstract class RestrictedBoltzmannMachineManager extends NetworkManager {

    protected final long seed;

    public RestrictedBoltzmannMachineManager(NeuralNetwork neuralNetwork, Database database, long seed) {
        super(neuralNetwork, database);
        this.seed = seed;
    }

    @Override
    public void recalculateHeuristic() {
        throw new UnsupportedOperationException("Not implemented yet");
    }
}
