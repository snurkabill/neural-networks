package net.snurkabill.neuralnetworks.newdata.database;

import net.snurkabill.neuralnetworks.math.function.transferfunction.TransferFunctionCalculator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Enumeration;

public class ClassificationDatabase extends NewDatabase {

    private static final Logger LOGGER = LoggerFactory.getLogger(ClassificationDatabase.class.getName());

    private final NewDataItem[][] trainingSet;
    private final NewDataItem[][] validationSet;
    private final NewDataItem[][] testingSet;

    public ClassificationDatabase(NewDataItem[][] trainingSet, NewDataItem[][] validationSet,
                                  NewDataItem[][] testingSet, TransferFunctionCalculator calculator,
                                  String databaseName, long seed) {
        super(calculateSetSize(trainingSet), calculateSetSize(validationSet), calculateSetSize(testingSet),
                trainingSet[0][0], databaseName, seed);
        if(trainingSet[0][0].getTargetSize() != trainingSet.length) {
            throw new IllegalArgumentException("Size of target vector doesn't correspond with count" +
                    "of classes");
        }
        this.trainingSet = trainingSet;
        this.validationSet = validationSet;
        this.testingSet = testingSet;
        alternateTarget(this.trainingSet, calculator);
        alternateTarget(this.validationSet, calculator);
        alternateTarget(this.testingSet, calculator);
        LOGGER.info("Database created.");
    }

    private void alternateTarget(NewDataItem[][] set, TransferFunctionCalculator calculator) {
        double lowLimit = calculator.getLowLimit();
        double topLimit = calculator.getTopLimit();
        for (int i = 0; i < set.length; i++) {
            for (int j = 0; j < set[i].length; j++) {
                for (int k = 0; k < this.getTargetSize(); k++) {
                    set[i][j].target[k] = set[i][j].target[k] == 0.0 ? lowLimit: topLimit;
                }
            }
        }
    }

    private static int calculateSetSize(NewDataItem[][] set) {
        int count = 0;
        for (NewDataItem[] aSet : set) {
            count += aSet.length;
        }
        return count;
    }

    public int getSizeOfTrainSetClass(int i) {
        return trainingSet[i].length;
    }

    public int getSizeOfValidationSetClass(int i) {
        return validationSet[i].length;
    }

    public int getSizeOfTestSetClass(int i) {
        return testingSet[i].length;
    }

    @Override
    public Enumeration<NewDataItem> createTrainingSetEnumerator() {
        return new EpochSimpleTrainingEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createInfiniteSimpleTrainingSetEnumerator() {
        return new InfiniteSimpleTrainingEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createInfiniteRandomTrainingSetEnumerator() {
        return new InfiniteRandomTrainingEnumerator();
    }

    @Override
    public Enumeration<NewDataItem> createValidationSetEnumerator() {
        return new ValidationSetEnumerator();
    }

    public Enumeration<NewDataItem> createValidationClassEnumerator(int _class) {
        return new ValidationClassEnumerator(_class);
    }

    @Override
    public Enumeration<NewDataItem> createTestingSetEnumerator() {
        return new TestingSetEnumerator();
    }

    public Enumeration<NewDataItem> createTestClassEnumerator(int _class) {
        return new TestClassEnumerator(_class);
    }

    public Enumeration<NewDataItem> createTrainingClassEnumerator(int _class) {
        return new TrainingClassEnumerator(_class);
    }

    public class InfiniteRandomTrainingEnumerator implements Enumeration<NewDataItem> {

        @Override
        public boolean hasMoreElements() {
            return true;
        }

        @Override
        public NewDataItem nextElement() {
            int randomIndex = random.nextInt(trainingSetSize);
            int tmpIndex = 0;
            int previousTmpIndex = 0;
            for (int i = 0; i < targetSize; i++) {
                tmpIndex += trainingSet[i].length;
                if(randomIndex < tmpIndex) {
                    randomIndex -= previousTmpIndex;
                    return trainingSet[i][randomIndex];
                }
                previousTmpIndex = tmpIndex;
            }
            throw new IllegalStateException("Index should have been picked");
        }
    }

    public abstract class SimpleTrainingEnumerator implements Enumeration<NewDataItem> {
        private final int[] lastUnused = new int[targetSize];
        private int lastUnusedClass;
        private int lastUsedClass;

        public SimpleTrainingEnumerator() {
            lastUnusedClass = 0;
            for (int i = 0; i < lastUnused.length; i++) {
                lastUnused[i] = trainingSet[i].length - 1;
            }
        }

        @Override
        public NewDataItem nextElement() {
            lastUnusedClass = (lastUnusedClass + 1) % targetSize;
            if (lastUnusedClass == 0) {
                lastUsedClass = trainingSet.length - 1;
            } else {
                lastUsedClass = lastUnusedClass - 1;
            }
            lastUnused[lastUsedClass] = (lastUnused[lastUsedClass] + 1) % trainingSet[lastUsedClass].length;
            if (lastUnused[lastUsedClass] == trainingSet[lastUsedClass].length) {
                lastUnused[lastUsedClass] = 0;
            }
            return trainingSet[lastUsedClass][lastUnused[lastUsedClass]];
        }

        public int nextClass() {
            return lastUnusedClass;
        }

        public int previousClass() {
            return lastUsedClass;
        }
    }

    public class InfiniteSimpleTrainingEnumerator extends SimpleTrainingEnumerator {

        @Override
        public boolean hasMoreElements() {
            return true;
        }
    }

    public class EpochSimpleTrainingEnumerator extends SimpleTrainingEnumerator {

        private int counter = 0;

        @Override
        public boolean hasMoreElements() {
            return counter < trainingSetSize;
        }

        @Override
        public NewDataItem nextElement() {
            NewDataItem item = super.nextElement();
            counter++;
            return item;
        }
    }

    public abstract class ClassEnumerator implements Enumeration<NewDataItem> {

        protected final int _class;
        protected int index;

        public ClassEnumerator(int _class) {
            this._class = _class;
            this.index = 0;
        }
    }

    public class TrainingClassEnumerator extends ClassEnumerator {

        public TrainingClassEnumerator(int _class) {
            super(_class);
        }

        @Override
        public boolean hasMoreElements() {
            return index < trainingSet[_class].length;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return trainingSet[_class][index - 1];
        }
    }

    public class TestClassEnumerator extends ClassEnumerator {

        public TestClassEnumerator(int _class) {
            super(_class);
        }

        @Override
        public boolean hasMoreElements() {
            return index < testingSet[_class].length;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return testingSet[_class][index - 1];
        }
    }

    public class ValidationClassEnumerator extends ClassEnumerator {

        public ValidationClassEnumerator(int _class) {
            super(_class);
        }

        @Override
        public boolean hasMoreElements() {
            return index < validationSet[_class].length;
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            return validationSet[_class][index - 1];
        }
    }

    public class TestingSetEnumerator implements Enumeration<NewDataItem> {

        private int _class;
        private int index;

        public TestingSetEnumerator() {
            this._class = 0;
            this.index = 0;
        }

        @Override
        public boolean hasMoreElements() {
            if(_class < targetSize - 1) {
                return true;
            } else {
                return index < testingSet[_class].length;
            }
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            NewDataItem item = testingSet[_class][index - 1];

            if(!(index < testingSet[_class].length)) {
                _class++;
                if(_class < testingSet.length){
                    index = 0;
                }
            }
            return item;
        }
    }

    public class ValidationSetEnumerator implements Enumeration<NewDataItem> {

        private int _class;
        private int index;

        public ValidationSetEnumerator() {
            this._class = 0;
            this.index = 0;
        }

        @Override
        public boolean hasMoreElements() {
            if(_class < targetSize - 1) {
                return true;
            } else {
                return index < validationSet[_class].length;
            }
        }

        @Override
        public NewDataItem nextElement() {
            index++;
            NewDataItem item = validationSet[_class][index - 1];

            if(!(index < validationSet[_class].length)) {
                _class++;
                if(_class < validationSet.length){
                    index = 0;
                }
            }
            return item;
        }
    }

}
