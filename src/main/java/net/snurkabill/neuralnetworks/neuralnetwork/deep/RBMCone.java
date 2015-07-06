package net.snurkabill.neuralnetworks.neuralnetwork.deep;

import net.snurkabill.neuralnetworks.managers.deep.GreedyLayerWiseDeepBeliefNetworkBuilder;
import net.snurkabill.neuralnetworks.neuralnetwork.FeedForwardable;
import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;
import net.snurkabill.neuralnetworks.utilities.DeepNetUtils;

import java.util.List;

public class RBMCone implements FeedForwardable {

    private final List<RestrictedBoltzmannMachine> machines;
    private final List<GreedyLayerWiseDeepBeliefNetworkBuilder.NormalizerVector> normalizers;

    public RBMCone(List<RestrictedBoltzmannMachine> machines,
                   List<GreedyLayerWiseDeepBeliefNetworkBuilder.NormalizerVector> normalizers) {
        this.machines = machines;
        DeepNetUtils.checkMachineList(machines);
        this.normalizers = normalizers;
    }

    public int getSizeOfInputVector() {
        return machines.get(0).getSizeOfInputVector();
    }

    @Override
    public void feedForward(double[] inputVector) {
        machines.get(0).setVisibleNeurons(inputVector);
        for (int k = 0; k < machines.size() - 1; k++) {
            machines.get(k + 1).setVisibleNeurons(normalizers.get(k).normalize(machines.get(k).activateHiddenNeurons()));
        }
        machines.get(machines.size() - 1).activateHiddenNeurons();
    }

    private void feedForward(double[] vector, int stoppingLayerIndex) {
        machines.get(0).setVisibleNeurons(vector);
        for (int k = 0; k < stoppingLayerIndex; k++) {
            machines.get(k + 1).setVisibleNeurons(normalizers.get(k).normalize(machines.get(k).activateHiddenNeurons()));
        }
    }

    private double[] feedForwardAndActivateLastHidden(double[] vector, int stoppingLayerIndex) {
        feedForward(vector, stoppingLayerIndex);
        return machines.get(stoppingLayerIndex).activateHiddenNeurons();
    }

    @Override
    public double[] getForwardedValues() {
        return machines.get(machines.size() - 1).getForwardedValues();
    }
}
