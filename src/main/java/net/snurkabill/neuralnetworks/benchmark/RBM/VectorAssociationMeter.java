package net.snurkabill.neuralnetworks.benchmark.RBM;

import net.snurkabill.neuralnetworks.benchmark.RBM.Vector.Tuple;

public class VectorAssociationMeter {

    public static double check(Tuple originalVector, Tuple associatedVector, boolean [] knownIndexes) {
        if(originalVector.getLength() != associatedVector.getLength() ||
                originalVector.getLength()!= knownIndexes.length) {
            throw new IllegalArgumentException("Arrays have different sizes");
        }
        int differentUnknownIndexes = 0;
        for (int i = 0; i < originalVector.getLength(); i++) {
            if(originalVector.getArray()[i] != associatedVector.getArray()[i]) {
                if(knownIndexes[i] != true) {
                    differentUnknownIndexes++;
                }
            }
        }
        return differentUnknownIndexes;
    }
}
