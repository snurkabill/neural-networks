package net.snurkabill.neuralnetworks.data.database;

import net.snurkabill.neuralnetworks.utilities.Utilities;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class Database<T extends DataItem> {

    private static final Logger LOGGER = LoggerFactory.getLogger("Database");

    private final Random random;

    private final DataItem[][] trainingSet;
    private final DataItem[][] testingSet;

    private final int sizeOfTrainingSet;
    private final int sizeOfTestingSet;

    private final double[] probabilitiesOfClasses;

    private final int numberOfClasses;
    private final String name;
    private final int sizeOfVector;

    public Database(long seed, Map<Integer, List<T>> trainingSet, Map<Integer, List<T>> testingSet,
                    String name, boolean applyFilter) {
        if(name == null) {
            throw new IllegalArgumentException("Database name cannot be null");
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
        LOGGER.info("Creating database {}", name);
        this.name = name;
        this.random = new Random(seed);
        boolean[] filter = makeFilter(applyFilter, trainingSet);
        this.trainingSet = applyFilterOnDatasetForDimReduction(filter, trainingSet);
        this.testingSet = applyFilterOnDatasetForDimReduction(filter, testingSet);
        this.sizeOfVector = this.trainingSet[0][0].data.length;
        this.numberOfClasses = this.trainingSet.length;
        this.sizeOfTrainingSet = sumElements(this.trainingSet);
        this.sizeOfTestingSet = sumElements(this.testingSet);
        this.probabilitiesOfClasses = new double[this.numberOfClasses];
        this.calculateProbabilitiesOfClasses();
        LOGGER.info("Database {} created!", name);
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
        LOGGER.info("Making filter for removing redundant dimensions on dataset");
        if (trainingDataset.isEmpty()) {
            throw new IllegalArgumentException("Training dataset is empty!");
        }
        for (Map.Entry<Integer, List<T>> list : trainingDataset.entrySet()) {
            if (list.getValue().isEmpty()) {
                throw new IllegalArgumentException("One class of dataset is empty!");
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
        LOGGER.info("Filter is done!");
        return isChanged;
    }

    private DataItem[][] applyFilterOnDatasetForDimReduction(boolean[] filter,
                                                                      Map<Integer, List<T>> dataSet) {
        LOGGER.info("Applying filter on data");
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
        LOGGER.info("Data are filtered!");
        return reducedData;
    }

    private void calculateProbabilitiesOfClasses() {
        for (int i = 0; i < this.trainingSet.length; i++) {
            probabilitiesOfClasses[i] = trainingSet[i].length / sizeOfTrainingSet;
        }
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
    public Database stochasticStandardNormalization(int percentageSampleSize) {
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

    public Database stochasticStandardNormalization(double scaleSize) {
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
        LOGGER.info("Normalizing finished");
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

    public Database stochasticUnityBasedNormalization(double scaleSize, double topLimit, double lowLimit) {
        checkScaleSize(scaleSize);
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
        LOGGER.info("Normalizing finished");
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

    @Deprecated
    public void storeDatabase() throws FileNotFoundException, IOException {
        LOGGER.info("Storing database: {}", name);
        long time1 = System.currentTimeMillis();
        try (DataOutputStream os = new DataOutputStream(new FileOutputStream(new File(name)))) {
            os.writeInt(numberOfClasses);
            os.writeInt(trainingSet[0][0].data.length);
            for (int i = 0; i < numberOfClasses; i++) {
                os.writeInt(trainingSet[i].length);
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < trainingSet[i].length; j++) {
                    for (int k = 0; k < trainingSet[0][0].data.length; k++) {
                        os.writeDouble(trainingSet[i][j].data[k]);
                    }
                }
            }
            for (int i = 0; i < numberOfClasses; i++) {
                os.writeInt(testingSet[i].length);
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < testingSet[i].length; j++) {
                    for (int k = 0; k < testingSet[0][0].data.length; k++) {
                        os.writeDouble(testingSet[i][j].data[k]);
                    }
                }
            }
            os.flush();
        }
        long time2 = System.currentTimeMillis();
        LOGGER.info("Storing database done, took: {}", time2 - time1);
    }

    @Deprecated
    public static Database loadDatabase(File baseFile) throws FileNotFoundException, IOException {
        return loadDatabase(baseFile, 0);
    }

    @Deprecated
    public static Database loadDatabase(File baseFile, long seed) throws FileNotFoundException, IOException {
        LOGGER.info("Loading database {}", baseFile.getName());
        long time1 = System.currentTimeMillis();
        Map<Integer, List<DataItem>> trainingSet;
        Map<Integer, List<DataItem>> testingSet;
        try (DataInputStream is = new DataInputStream(new FileInputStream(baseFile))) {
            trainingSet = new HashMap<>();
            testingSet = new HashMap<>();
            int numberOfClasses = is.readInt();
            for (int i = 0; i < numberOfClasses; i++) {
                trainingSet.put(i, new ArrayList<DataItem>());
                testingSet.put(i, new ArrayList<DataItem>());
            }
            int sizeOfVector = is.readInt();
            List<Integer> sizesOfTrainingLists = new ArrayList<>();
            for (int i = 0; i < numberOfClasses; i++) {
                sizesOfTrainingLists.add(is.readInt());
            }
            double[] buffer = new double[sizeOfVector];
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < sizesOfTrainingLists.get(i); j++) {
                    for (int k = 0; k < sizeOfVector; k++) {
                        buffer[k] = is.readDouble();
                    }
                    trainingSet.get(i).add(new DataItem(buffer));
                }
            }
            List<Integer> sizesOfTestingLists = new ArrayList<>();
            for (int i = 0; i < numberOfClasses; i++) {
                sizesOfTestingLists.add(is.readInt());
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < sizesOfTestingLists.get(i); j++) {
                    for (int k = 0; k < sizeOfVector; k++) {
                        buffer[k] = is.readDouble();
                    }
                    testingSet.get(i).add(new DataItem(buffer));
                }
            }
        }
        long time2 = System.currentTimeMillis();
        LOGGER.info("Loading database done, took: {}", time2 - time1);
        return new Database(seed, trainingSet, testingSet, baseFile.getName(), false);
    }

}
