package net.snurkabill.neuralnetworks.data.database;

import net.snurkabill.neuralnetworks.utilities.Timer;
import net.snurkabill.neuralnetworks.utilities.Utilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Random;

public class ClassFullDatabase<T extends DataItem> {

    private static final Logger LOGGER = LoggerFactory.getLogger("ClassFullDatabase");

    private final Timer timer = new Timer();
    private final Random random;

    private final DataItem[][] trainingSet;
    private final DataItem[][] testingSet;

    private final int sizeOfTrainingSet;
    private final int sizeOfTestingSet;

    private final int numberOfClasses;
    private final String name;
    private final int sizeOfVector;

    public ClassFullDatabase(long seed, File dataFile) {
        if(dataFile == null) {
            throw new IllegalArgumentException("Wrong file name: (" + dataFile + ")");
        }
        constructingDatabaseStarted(dataFile.getName());
        this.name = dataFile.getName();
        this.random = new Random(seed);
        try (DataInputStream inputStream = new DataInputStream(new FileInputStream(dataFile))) {
            this.numberOfClasses = inputStream.readInt();
            this.trainingSet = new DataItem[numberOfClasses][];
            this.testingSet = new DataItem[numberOfClasses][];
            this.sizeOfVector = inputStream.readInt();
            int tmpSizeOfSet = 0;
            for (int i = 0; i < numberOfClasses; i++) {
                this.trainingSet[i] = new DataItem[inputStream.readInt()];
                tmpSizeOfSet += trainingSet[i].length;
            }
            this.sizeOfTrainingSet = tmpSizeOfSet;
            double[] tmpArray = new double[sizeOfVector];
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < trainingSet[i].length; j++) {
                    for (int k = 0; k < this.sizeOfVector; k++) {
                        tmpArray[k] = inputStream.readDouble();
                    }
                    trainingSet[i][j] = new DataItem(tmpArray);
                }
            }
            tmpSizeOfSet = 0;
            for (int i = 0; i < numberOfClasses; i++) {
                this.testingSet[i] = new DataItem[inputStream.readInt()];
                tmpSizeOfSet += testingSet[i].length;
            }
            this.sizeOfTestingSet = tmpSizeOfSet;
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < testingSet[i].length; j++) {
                    for (int k = 0; k < sizeOfVector; k++) {
                        tmpArray[k] = inputStream.readDouble();
                    }
                    testingSet[i][j] = new DataItem(tmpArray);
                }
            }
        } catch (Exception e) {
            throw new RuntimeException("Loading classFullDatabase failed!", e);
        }
        constructingDatabaseEnded();
    }

    public ClassFullDatabase(long seed, Map<Integer, List<T>> trainingSet, Map<Integer, List<T>> testingSet,
                             String name, boolean applyFilter) {
        constructingDatabaseStarted(name);
        if(name == null) {
            throw new IllegalArgumentException("ClassFullDatabase name cannot be null");
        }
        if(testingSet == null) {
            throw new IllegalArgumentException("Testing set cannot be null");
        }
        if(trainingSet == null) {
            throw new IllegalArgumentException("Training set cannot be null");
        }
        if (trainingSet.size() != testingSet.size()) {
            throw new IllegalArgumentException("TrainingSet has different size than testingSet");
        }
        this.name = name;
        this.random = new Random(seed);
        boolean[] filter = makeFilter(applyFilter, trainingSet);
        LOGGER.info("Converting data");
        int previousVectorSize = trainingSet.get(0).get(0).data.length;
        this.timer.startTimer();
        this.trainingSet = convertDataToArrays(filter, trainingSet);
        this.testingSet = convertDataToArrays(filter, testingSet);
        LOGGER.info("Converting is done, took {} seconds", timer.secondsSpent());
        LOGGER.info("Old vector size: {}, new vector size: {}", previousVectorSize, this.trainingSet[0][0].data.length);
        this.sizeOfVector = this.trainingSet[0][0].data.length;
        this.numberOfClasses = this.trainingSet.length;
        this.sizeOfTrainingSet = sumElements(this.trainingSet);
        this.sizeOfTestingSet = sumElements(this.testingSet);
        constructingDatabaseEnded();
    }

    private void constructingDatabaseStarted(String name) {
        this.timer.startTimer();
        LOGGER.info("Creating classFullDatabase {}", name);
    }

    private void constructingDatabaseEnded() {
        LOGGER.info("ClassFullDatabase {} created after {} seconds", name, timer.secondsSpent());
    }

    public String getName() {
        return name;
    }

    private boolean[] makeFilter(boolean applyFilter, Map<Integer, List<T>> trainingSet) {
        int sizeOfVector = trainingSet.get(0).get(0).data.length;
        boolean[] filter = applyFilter ? makeFilterForRemovingRedundantDimensions(trainingSet) :
                new boolean[sizeOfVector];
        if (!applyFilter) {
            for (int i = 0; i < filter.length; i++) {
                filter[i] = true;
            }
        }
        return filter;
    }

    private int sumElements(DataItem[][] dataSet) {
        int sum = 0;
        for (int i = 0; i < this.trainingSet.length; i++) {
            sum += dataSet[i].length;
        }
        return sum;
    }

    private boolean[] makeFilterForRemovingRedundantDimensions(Map<Integer, List<T>> trainingDataset) {
        timer.startTimer();
        LOGGER.info("Making filter for removing redundant dimensions of dataSet");
        if (trainingDataset.isEmpty()) {
            throw new IllegalArgumentException("Training dataSet is empty!");
        }
        for (Map.Entry<Integer, List<T>> list : trainingDataset.entrySet()) {
            if (list.getValue().isEmpty()) {
                throw new IllegalArgumentException("One class of dataSet is empty!");
            }
        }
        int sizeOfVector = trainingDataset.get(0).get(0).data.length;
        double[] tmpSnapshot = new double[sizeOfVector];
        boolean[] isChanged = new boolean[sizeOfVector];
        System.arraycopy(trainingDataset.get(0).get(0).data, 0, tmpSnapshot, 0, tmpSnapshot.length);
        for (int i = 0; i < trainingDataset.size(); i++) {
            for (int j = 0; j < trainingDataset.get(i).size(); j++) {
                for (int k = 0; k < tmpSnapshot.length; k++) {
                    if (tmpSnapshot[k] != trainingDataset.get(i).get(j).data[k]) {
                        isChanged[k] = true;
                    }
                }
            }
        }
        LOGGER.info("Filter has been created after {} seconds", timer.secondsSpent());
        return isChanged;
    }

    private DataItem[][] convertDataToArrays(boolean[] filter,
                                             Map<Integer, List<T>> dataSet) {
        DataItem[][] reducedData = new DataItem[dataSet.size()][];
        int sizeOfNewVector = 0;
        for (boolean _filter : filter) {
            if (_filter) {
                sizeOfNewVector++;
            }
        }
        for (int i = dataSet.size() - 1; i >= 0; i--) {
            reducedData[i] = new DataItem[dataSet.get(i).size()];
            for (int j = reducedData[i].length - 1; j >= 0; j--) {
                double[] data = new double[sizeOfNewVector];
                for (int k = 0, move = 0; k < filter.length; k++) {
                    if(filter[k]) {
                        data[move] = dataSet.get(i).get(j).data[k];
						move++;
                    }
                }
                reducedData[i][(reducedData[i].length - 1) - j] = new DataItem(data);
                dataSet.get(i).remove(j);
            }
            dataSet.remove(i);
        }
        return reducedData;
    }

    private void checkScaleSize(double scaleSize) {
        if(scaleSize > 1) {
            throw new IllegalArgumentException("ScaleSize (" + scaleSize + ") is too large," +
                    " correct interval is (0, 1>");
        }
        if(scaleSize <= 0) {
            throw new IllegalArgumentException("ScaleSize (" + scaleSize + ") is too low, correct interval is (0, 1>");
        }
    }

    // TODO: implement deterministic normalization
    public ClassFullDatabase stochasticStandardNormalization(int percentageSampleSize) {
        if(percentageSampleSize <= 0) {
            throw new IllegalArgumentException("percentageSampleSize (" + percentageSampleSize + ") is too low. " +
                    "Must be from interval (0, 100>");
        }
        if(percentageSampleSize > 100) {
            throw new IllegalArgumentException("percentageSampleSize (" + percentageSampleSize + ") is too high. " +
                    "Must be from interval (0, 100>");
        }
        return stochasticStandardNormalization(percentageSampleSize / 100.0);
    }

    private int calculatePercentageSize(double scaleSize) {
        int tmp = (int) (sizeOfTrainingSet * scaleSize);
        if(tmp > sizeOfTrainingSet) {
            tmp = sizeOfTrainingSet;
        }
        return tmp;
    }

    public ClassFullDatabase stochasticStandardNormalization(double scaleSize) {
        this.timer.startTimer();
        checkScaleSize(scaleSize);
        LOGGER.info("Data standard normalization started.");
        for (int i = 0; i < this.getSizeOfVector(); i++) {
            int sampleSize = calculatePercentageSize(scaleSize);
            double[] array = new double[sampleSize];
            InfiniteRandomTrainingIterator iterator = this.getInfiniteRandomTrainingIterator();
            for (int j = 0; j < sampleSize; j++) {
                array[j] = iterator.next().data[i];
            }
            double mean = Utilities.mean(array);
            double stdev = Utilities.stddev(array, mean);
            applyStdNormalizationOnIthDimension(i, mean, stdev, trainingSet);
            applyStdNormalizationOnIthDimension(i, mean, stdev, testingSet);
        }
        LOGGER.info("Normalizing finished. Spent {} seconds", this.timer.secondsSpent());
        return this;
    }

    private void applyStdNormalizationOnIthDimension(int i, double mean, double stdev, DataItem[][] set) {
        for (DataItem[] _class : set) {
            for (DataItem item : _class) {
                if (stdev == 0) {
                    item.data[i] = 0.0;
                } else {
                    item.data[i] = (item.data[i] - mean) / stdev;
                }
            }
        }
    }

    public ClassFullDatabase stochasticUnityBasedNormalization(double scaleSize, double topLimit, double lowLimit) {
        checkScaleSize(scaleSize);
        this.timer.startTimer();
        LOGGER.info("Data unity-based normalization started.");
        for (int i = 0; i < this.getSizeOfVector(); i++) {
            int sampleSize = calculatePercentageSize(scaleSize);
            double max = Double.NEGATIVE_INFINITY;
            double min = Double.POSITIVE_INFINITY;
            InfiniteRandomTrainingIterator iterator = this.getInfiniteRandomTrainingIterator();
            for (int j = 0; j < sampleSize; j++) {
                double item = iterator.next().data[i];
                if(item > max) {
                    max = item;
                }
                if(item < min) {
                    min = item;
                }
            }
            applyUnitBasedNormalizationOnIthDimension(i, max, min, topLimit, lowLimit, trainingSet);
            applyUnitBasedNormalizationOnIthDimension(i, max, min, topLimit, lowLimit, testingSet);
        }
        LOGGER.info("Data unity-based normalization finished. Took {} seconds", this.timer.secondsSpent());
        return this;
    }

    private void applyUnitBasedNormalizationOnIthDimension(int i, double max, double min,double topLimit,
                                                           double lowLimit, DataItem[][] set) {
        for (DataItem[] _class : set) {
            for (DataItem item : _class) {
                item.data[i] = lowLimit + ((item.data[i] - min) * (topLimit - lowLimit) / (max - min));
            }
        }
    }

    public int getTestSetSize() {
        return sizeOfTestingSet;
    }

    public int getTrainingSetSize() {
        return sizeOfTrainingSet;
    }

    public int getNumberOfClasses() {
        return numberOfClasses;
    }

    public int getSizeOfTestingDataset(int index) {
        return testingSet[index].length;
    }

    public int getSizeOfTrainingDataset(int index) {
        return trainingSet[index].length;
    }

    public int getSizeOfVector() {
        return sizeOfVector;
    }

    public synchronized TestClassIterator getTestingIteratorOverClass(int _class) {
        return new TestClassIterator(_class);
    }

    public synchronized TrainingClassIterator getTrainingIteratorOverClass(int _class) {
        return new TrainingClassIterator(_class);
    }

    public synchronized InfiniteSimpleTrainingIterator getInfiniteTrainingIterator() {
        return new InfiniteSimpleTrainingIterator();
    }

    public synchronized InfiniteRandomTrainingIterator getInfiniteRandomTrainingIterator() {
        return new InfiniteRandomTrainingIterator();
    }

    public class InfiniteRandomTrainingIterator implements Iterator<LabelledItem> {

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public LabelledItem next() {
            int randomIndex = random.nextInt(sizeOfTrainingSet);
            int tmpIndex = 0;
            int previousTmpIndex = 0;
            for (int i = 0; i < numberOfClasses; i++) {
                tmpIndex += trainingSet[i].length;
                if(randomIndex < tmpIndex) {
                    randomIndex -= previousTmpIndex;
                    return new LabelledItem(trainingSet[i][randomIndex], i);
                }
                previousTmpIndex = tmpIndex;
            }
            throw new IllegalStateException("Index should have been picked");
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    public class InfiniteSimpleTrainingIterator implements Iterator<LabelledItem> {

        private final int[] lastUnused = new int[getNumberOfClasses()];
        private int lastUnusedClass;
        private int lastUsedClass;

        public InfiniteSimpleTrainingIterator() {
            lastUnusedClass = 0;
            for (int i = 0; i < lastUnused.length; i++) {
                lastUnused[i] = trainingSet[i].length - 1;
            }
        }

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public LabelledItem next() {
            lastUnusedClass = (lastUnusedClass + 1) % numberOfClasses;
            if (lastUnusedClass == 0) {
                lastUsedClass = trainingSet.length - 1;
            } else {
                lastUsedClass = lastUnusedClass - 1;
            }
            lastUnused[lastUsedClass] = (lastUnused[lastUsedClass] + 1) % trainingSet[lastUsedClass].length;
            if (lastUnused[lastUsedClass] == trainingSet[lastUsedClass].length) {
                lastUnused[lastUsedClass] = 0;
            }
            return new LabelledItem(trainingSet[lastUsedClass][lastUnused[lastUsedClass]], lastUsedClass);
        }

        public int nextClass() {
            return lastUnusedClass;
        }

        public int previousClass() {
            return lastUsedClass;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    public class TestClassIterator implements Iterator<LabelledItem> {

        private final int _class;
        private int index;

        public TestClassIterator(int _class) {
            this._class = _class;
            index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < testingSet[_class].length;
        }

        @Override
        public LabelledItem next() {
            index++;
            return new LabelledItem(testingSet[_class][index - 1], _class);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    public class TrainingClassIterator implements Iterator<LabelledItem> {

        private final int _class;
        private int index;

        public TrainingClassIterator(int _class) {
            this._class = _class;
            index = 0;
        }

        @Override
        public boolean hasNext() {
            return index < trainingSet[_class].length;
        }

        @Override
        public LabelledItem next() {
            index++;
            return new LabelledItem(trainingSet[_class][index - 1], _class);
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    public void storeDatabase() {
        this.timer.startTimer();
        LOGGER.info("Storing classFullDatabase started with ({}) filename", this.name);
        try (DataOutputStream outputStream = new DataOutputStream(new FileOutputStream(new File(this.name)))) {
            outputStream.writeInt(numberOfClasses);
            outputStream.writeInt(trainingSet[0][0].data.length);
            for (int i = 0; i < numberOfClasses; i++) {
                outputStream.writeInt(trainingSet[i].length);
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < trainingSet[i].length; j++) {
                    for (int k = 0; k < trainingSet[0][0].data.length; k++) {
                        outputStream.writeDouble(trainingSet[i][j].data[k]);
                    }
                }
            }
            for (int i = 0; i < numberOfClasses; i++) {
                outputStream.writeInt(testingSet[i].length);
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < testingSet[i].length; j++) {
                    for (int k = 0; k < testingSet[0][0].data.length; k++) {
                        outputStream.writeDouble(testingSet[i][j].data[k]);
                    }
                }
            }
            outputStream.flush();
        } catch (Exception e) {
            throw new RuntimeException("Storing classFullDatabase failed!", e);
        }
        LOGGER.info("Storing classFullDatabase is done after {} seconds", timer.secondsSpent());
    }

}
