package net.snurkabill.neuralnetworks.utilities;

import net.snurkabill.neuralnetworks.neuralnetwork.energybased.boltzmannmodel.restrictedboltzmannmachine.RestrictedBoltzmannMachine;

import java.util.List;

public class DeepNetUtils {

    public static void checkMachineList(List<RestrictedBoltzmannMachine> machines) {
        if (machines == null) {
            throw new IllegalArgumentException("List of machines is null");
        }
        if (machines.isEmpty()) {
            throw new IllegalArgumentException("List of RBMs is empty");
        }
        for (int i = 0; i < machines.size() - 1; i++) {
            if (machines.get(i).getSizeOfHiddenVector() != machines.get(i + 1).getSizeOfVisibleVector()) {
                throw new IllegalStateException("(" + i + " )th rbm has different hidden vector size than " +
                        "( " + (i + 1) + ")th rbm");
            }
        }
    }
}
