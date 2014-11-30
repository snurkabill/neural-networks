package net.snurkabill.neuralnetworks.managers.boltzmannmodel;

import net.snurkabill.neuralnetworks.data.database.Database;
import net.snurkabill.neuralnetworks.heuristic.calculators.HeuristicCalculator;
import net.snurkabill.neuralnetworks.managers.NetworkManager;
import net.snurkabill.neuralnetworks.neuralnetwork.NeuralNetwork;

public abstract class RestrictedBoltzmannMachineManager extends NetworkManager {

    protected final long seed;

    public RestrictedBoltzmannMachineManager(NeuralNetwork neuralNetwork, Database database, long seed,
                                             HeuristicCalculator heuristicCalculator) {
        super(neuralNetwork, database, heuristicCalculator);
        this.seed = seed;
    }

    @Override
    protected void checkVectorSizes() {
        if (neuralNetwork.getSizeOfInputVector() != database.getNumberOfClasses() + database.getSizeOfVector()) {
            throw new IllegalArgumentException("Size of input vector is different from number of classes");
        }
    }

}
