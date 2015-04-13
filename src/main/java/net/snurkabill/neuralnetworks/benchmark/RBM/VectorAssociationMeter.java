package net.snurkabill.neuralnetworks.benchmark.RBM;

import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.Tuple;

public class VectorAssociationMeter {

    public static double check(Tuple originalVector, Tuple associatedVector, boolean[] knownIndexes, int numOfUnknown) {
        if (originalVector.getLength() != associatedVector.getLength() ||
                originalVector.getLength() != knownIndexes.length) {
            throw new IllegalArgumentException("Arrays have different sizes");
        }
        int differentUnknownIndexes = 0;
        for (int i = 0; i < originalVector.getLength(); i++) {
            if (originalVector.getArray()[i] != associatedVector.getArray()[i]) {
                if (knownIndexes[i]) {
                    differentUnknownIndexes++;
                }
            }
        }
        double check = ((double) differentUnknownIndexes / (double) numOfUnknown) * 100.0;
        if (check > 100) {
            throw new IllegalArgumentException("KURVA CO TO DOPRDELE JE" +
                    "DifferentUnknownIndexes: " + differentUnknownIndexes +
                    "numOfUnknown: " + numOfUnknown);
        }
        return check;
    }
}
