package net.snurkabill.neuralnetworks.newdata.database;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Enumeration;

public class RegressionDatabase extends NewDatabase {

    private static final Logger LOGGER = LoggerFactory.getLogger(RegressionDatabase.class.getName());

    private final NewDataItem[] trainingSet;
    private final NewDataItem[] validationSet;
    private final NewDataItem[] testingSet;

    public RegressionDatabase(NewDataItem[] trainingData, NewDataItem[] validationData, NewDataItem[] testingData,
                              String databaseName, long seed) {
        super(trainingData.length, validationData.length, testingData.length, checkAndPropagate(), databaseName, seed);
        this.trainingSet = trainingData;
        this.validationSet = validationData;
        this.testingSet = testingData;
        LOGGER.info("Database created.");
    }

    public static NewDataItem checkAndPropagate(NewDataItem trainingData, NewDataItem validationData,
                                              NewDataItem testingData) {
        if(trainingData.getDataVectorSize() != validationData.getDataVectorSize() ||
                trainingData.getDataVectorSize() != testingData.getDataVectorSize()) {
            throw new IllegalArgumentException("Data from datasets have different size of data vector");
        }
        if(trainingData.getTargetSize() != validationData.getTargetSize() ||
                trainingData.getTargetSize() != testingData.getTargetSize()) {
            throw new IllegalArgumentException("Data from datasets have different size of target vector");
        }
        return trainingData;
    }

    @Override
    public Enumeration<NewDataItem> createTrainingSetEnumerator() {
        return new EpochTrainingSetEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createInfiniteSimpleTrainingSetEnumerator() {
        return new InfiniteSimpleTrainingSetEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createInfiniteRandomTrainingSetEnumerator() {
        return new InfiniteRandomTrainingSetEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createValidationSetEnumerator() {
        return new ValidationSetEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createTestingSetEnumerator() {
        return new TestingSetEnumerator();
    }

    public class InfiniteRandomTrainingSetEnumerator implements Enumeration<NewDataItem> {

        @Override
        public boolean hasMoreElements() {
            return true;
        }

        @Override
        public NewDataItem nextElement() {
            return trainingSet[random.nextInt(trainingSetSize)];
        }
    }

    public class InfiniteSimpleTrainingSetEnumerator implements Enumeration<NewDataItem> {

        private int index = 0;

        @Override
        public boolean hasMoreElements() {
            return true;
        }

        @Override
        public NewDataItem nextElement() {
            if(index == trainingSetSize) {
                index = 0;
            }
            index++;
            return trainingSet[index - 1];
        }
    }

    public class EpochTrainingSetEnumerator implements Enumeration<NewDataItem> {

        private int index = 0;

        @Override
        public boolean hasMoreElements() {
            return index < trainingSetSize;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return trainingSet[index - 1];
        }
    }

    public class ValidationSetEnumerator implements Enumeration<NewDataItem> {

        private int index = 0;

        @Override
        public boolean hasMoreElements() {
            return index < validationSetSize;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return validationSet[index - 1];
        }
    }

    public class TestingSetEnumerator implements Enumeration<NewDataItem> {

        private int index = 0;

        @Override
        public boolean hasMoreElements() {
            return index < testingSetSize;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return testingSet[index - 1];
        }
    }

}
