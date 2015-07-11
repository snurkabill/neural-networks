package net.snurkabill.neuralnetworks.newdata.database;

import net.snurkabill.neuralnetworks.utilities.Timer;

import java.util.Enumeration;
import java.util.Random;

public abstract class NewDatabase<T extends NewDataItem> {

    protected final String databaseName;
    protected final Timer timer = new Timer();
    protected final Random random;

    protected final int trainingSetSize;
    protected final int validationSetSize;
    protected final int testingSetSize;

    protected final boolean hasTarget;
    protected final int vectorSize;
    protected final int targetSize;

    public NewDatabase(int trainingSetSize, int validationSetSize, int testingSetSize,
                       NewDataItem item, String databaseName, long seed) {
        if(trainingSetSize <= 0) {
            throw new IllegalArgumentException("Size of TrainingSet must be at least one");
        }
        if(validationSetSize <= 0) {
            throw new IllegalArgumentException("Size of ValidationSet must be at least one");
        }
        if(testingSetSize <= 0) {
            throw new IllegalArgumentException("Size of TestingSet must be at least one");
        }
        if(item == null) {
            throw new IllegalArgumentException("Item can't be null");
        }
        this.random = new Random(seed);
        this.trainingSetSize = trainingSetSize;
        this.validationSetSize = validationSetSize;
        this.testingSetSize = testingSetSize;
        this.databaseName = databaseName;
        this.vectorSize = item.getDataVectorSize();
        if(item.hasTarget()) {
            this.targetSize = item.getTargetSize();
        } else {
            this.targetSize = 0;
        }
        this.hasTarget = targetSize != 0;
    }

    public int getTrainingSetSize() {
        return trainingSetSize;
    }

    public int getValidationSetSize() {
        return validationSetSize;
    }

    public int getTestingSetSize() {
        return testingSetSize;
    }

    public String getDatabaseName() {
        return databaseName;
    }

    public boolean hasTarget() {
        return hasTarget;
    }

    public int getVectorSize() {
        return vectorSize;
    }

    public int getTargetSize() {
        return targetSize;
    }

    public abstract Enumeration<NewDataItem> createTrainingSetEnumerator();

    public abstract Enumeration<NewDataItem> createInfiniteSimpleTrainingSetEnumerator();

    public abstract Enumeration<NewDataItem> createInfiniteRandomTrainingSetEnumerator();

    public abstract Enumeration<NewDataItem> createValidationSetEnumerator();

    public abstract Enumeration<NewDataItem> createTestingSetEnumerator();

}
