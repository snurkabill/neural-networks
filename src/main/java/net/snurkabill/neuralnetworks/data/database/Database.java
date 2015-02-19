package net.snurkabill.neuralnetworks.data.database;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;

public class Database<T extends DataItem> {

    private static final Logger LOGGER = LoggerFactory.getLogger("Database");

    private final Random random;

    private final Map<Integer, List<T>> trainingSet;
    private final Map<Integer, List<T>> testingSet;
    private final Map<Integer, Iterator<T>> trainingSetIterator;
    private final Map<Integer, Iterator<T>> testingSetIterator;

    private int sizeOfTrainingSet;
    private int sizeOfTestingSet;
    private int numberOfClasses;
    private final String name;
    private final int sizeOfVector;

    public Database(long seed, Map<Integer, List<T>> trainingSet, Map<Integer, List<T>> testingSet,
                    String name, boolean applyFilter) {
        if (trainingSet.size() != testingSet.size()) {
            throw new IllegalArgumentException("TrainingSet has different size than testingSet");
        }
        LOGGER.info("Creating database {} ...", name);
        this.name = name;
        this.random = new Random();
        this.random.setSeed(seed);
        boolean[] filter = applyFilter ? makeFilterForRemovingRedundantDimensions(trainingSet) :
                new boolean[trainingSet.get(0).get(0).data.length];
        if (!applyFilter) {
            for (int i = 0; i < filter.length; i++) {
                filter[i] = true;
            }
        }
        this.trainingSet = applyFilterOnDatasetForDimReduction(filter, trainingSet);
        this.testingSet = applyFilterOnDatasetForDimReduction(filter, testingSet);
        this.numberOfClasses = this.trainingSet.size();
        this.testingSetIterator = new HashMap<>(this.trainingSet.size(), 1);
        this.trainingSetIterator = new HashMap<>(this.trainingSet.size(), 1);
        for (int i = 0; i < this.trainingSet.size(); i++) {
            sizeOfTrainingSet += this.trainingSet.get(i).size();
            sizeOfTestingSet += this.testingSet.get(i).size();
            testingSetIterator.put(i, this.testingSet.get(i).iterator());
            trainingSetIterator.put(i, this.trainingSet.get(i).iterator());
        }
        sizeOfVector = this.trainingSet.get(0).get(0).data.length;
        LOGGER.info("Database {} created!", name);
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
        double[] tmpSnapshot = new double[trainingDataset.get(0).get(0).data.length];
        boolean[] isChanged = new boolean[trainingDataset.get(0).get(0).data.length];
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


    private Map<Integer, List<T>> applyFilterOnDatasetForDimReduction(boolean[] filter,
                                                                      Map<Integer, List<T>> dataSet) {
        LOGGER.info("Applying filter on data");
        Map<Integer, List<T>> reducedData = new LinkedHashMap<>(dataSet.size(), 1);
        int sizeOfNewVector = 0;
        for (boolean _filter : filter) {
            if (_filter) {
                sizeOfNewVector++;
            }
        }
        for (int i = dataSet.size() - 1; i >= 0; i--) {
            List<T> tmpList = new ArrayList<>(dataSet.get(i).size());
            for (int j = dataSet.get(i).size() - 1; j >= 0; j--) {
                double[] data = new double[sizeOfNewVector];
                for (int k = 0, move = 0; k < filter.length; k++) {
                    if (filter[k]) {
                        data[move] = dataSet.get(i).get(j).data[k];
                        move++;
                    }
                }
                DataItem item = new DataItem(data);
                tmpList.add((T) item);
                dataSet.get(i).remove(j);
            }
            dataSet.remove(i);
            reducedData.put(i, tmpList);
        }
        LOGGER.info("Data are filtered!");
        return reducedData;
    }

    public T getTrainingIteratedData(int i) {
        if (!trainingSetIterator.get(i).hasNext()) {
            LOGGER.debug("Setting new iterator of trainingSet for {}th list", i);
            trainingSetIterator.put(i, trainingSet.get(i).iterator());
        }
        return trainingSetIterator.get(i).next();
    }

    public T getTestingIteratedData(int i) {
        if (!testingSetIterator.get(i).hasNext()) {
            LOGGER.debug("Setting new iterator of testingSet for {}th list", i);
            testingSetIterator.put(i, testingSet.get(i).iterator());
        }
        return testingSetIterator.get(i).next();
    }

    private T getTrainingData(int i) {
        List<T> list = trainingSet.get(i);
        return list.get(random.nextInt(list.size()));
    }

    private T getTestingData(int i) {
        List<T> list = testingSet.get(i);
        return list.get(random.nextInt(list.size()));
    }

    public LabelledItem getRandomizedLabelTestingData() {
        int _class = random.nextInt(testingSet.size());
        return new LabelledItem(getTestingData(_class), _class);
    }

    public LabelledItem getRandomizedLabelTrainingData() {
        int _class = random.nextInt(trainingSet.size());
        return new LabelledItem(getTrainingData(_class), _class);
    }

    public T getRandomizedTestingData() {
        return getTestingData(random.nextInt(testingSet.size()));
    }

    public T getRandomizedTrainingData() {
        return getTrainingData(random.nextInt(trainingSet.size()));
    }

    public int getTestSetSize() {
        return sizeOfTestingSet;
    }

    public int getTrainingsetSize() {
        return sizeOfTrainingSet;
    }

    public int getNumberOfClasses() {
        return numberOfClasses;
    }

    public int getSizeOfTestingDataset(int index) {
        return testingSet.get(index).size();
    }

    public int getSizeOfTrainingDataset(int index) {
        return trainingSet.get(index).size();
    }

    public int getSizeOfVector() {
        return sizeOfVector;
    }

    @Deprecated
    public void storeDatabase() throws FileNotFoundException, IOException {
        LOGGER.info("Storing database: {}", name);
        long time1 = System.currentTimeMillis();
        try (DataOutputStream os = new DataOutputStream(new FileOutputStream(new File(name)))) {
            os.writeInt(numberOfClasses);
            os.writeInt(trainingSet.get(0).get(0).data.length);
            for (int i = 0; i < numberOfClasses; i++) {
                os.writeInt(trainingSet.get(i).size());
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < trainingSet.get(i).size(); j++) {
                    for (int k = 0; k < trainingSet.get(0).get(0).data.length; k++) {
                        os.writeDouble(trainingSet.get(i).get(j).data[k]);
                    }
                }
            }
            for (int i = 0; i < numberOfClasses; i++) {
                os.writeInt(testingSet.get(i).size());
            }
            for (int i = 0; i < numberOfClasses; i++) {
                for (int j = 0; j < testingSet.get(i).size(); j++) {
                    for (int k = 0; k < testingSet.get(0).get(0).data.length; k++) {
                        os.writeDouble(testingSet.get(i).get(j).data[k]);
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

    public synchronized Iterator<T> getTestingIteratorOverClass(int i) {
        return testingSet.get(i).iterator();
    }

    public synchronized Iterator<T> getTrainingIteratorOverClass(int i) {
        return trainingSet.get(i).iterator();
    }

    public synchronized Iterator<T> getTestingIterator() {
        return new TestSetIterator(testingSet);
    }

    public synchronized InfiniteSimpleTrainSetIterator getInfiniteTrainingIterator() {
        return new InfiniteSimpleTrainSetIterator(trainingSet);
    }

    public synchronized InfiniteRandomTrainSetIterator getInfiniteRandomTrainingIterator() {
        return new InfiniteRandomTrainSetIterator();
    }

    public class InfiniteRandomTrainSetIterator implements Iterator {

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public LabelledItem next() {
            int randomNum = random.nextInt(trainingSet.size());
            LabelledItem item = new LabelledItem(getTrainingData(randomNum), randomNum);
            return item;
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }

    public class InfiniteSimpleTrainSetIterator implements Iterator {

        private final Map<Integer, List<T>> map;
        private final List<Iterator<T>> iterators = new ArrayList<>();
        private int lastUnusedClass;
        private int lastUsedClass;

        public InfiniteSimpleTrainSetIterator(Map<Integer, List<T>> map) {
            this.map = map;
            for (int i = 0; i < map.size(); i++) {
                iterators.add(map.get(i).iterator());
            }
            lastUnusedClass = 0;
            lastUsedClass = 0;
            // FINISH ME
        }

        @Override
        public boolean hasNext() {
            return true;
        }

        @Override
        public LabelledItem next() {
            if (lastUnusedClass == map.size()) {
                lastUnusedClass = 0;
            }
            if (!iterators.get(lastUnusedClass).hasNext()) {
                iterators.set(lastUnusedClass, map.get(lastUnusedClass).iterator());
            }
            lastUnusedClass++;
            if (lastUnusedClass == 0) {
                lastUsedClass = map.size() - 1;
            } else {
                lastUsedClass = lastUnusedClass - 1;
            }
            return new LabelledItem(iterators.get(lastUnusedClass - 1).next().data, lastUnusedClass - 1);
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

    public class TestSetIterator implements Iterator {
        private final List<Iterator<T>> iterators;
        private final Iterator<Iterator<T>> master;
        private Iterator<T> actual;
        private int actualClass;

        public TestSetIterator(Map<Integer, List<T>> map) {
            iterators = new ArrayList<>(map.size());
            for (int i = 0; i < map.size(); i++) {
                iterators.add(map.get(i).iterator());
            }
            master = iterators.iterator();
            actual = master.next();
            actualClass = 0;
        }

        @Override
        public boolean hasNext() {
            return master.hasNext() || actual.hasNext();
        }

        @Override
        public LabelledItem next() {
            if (actual.hasNext()) {
                return new LabelledItem(actual.next(), actualClass);
            } else {
                actual = master.next();
                actualClass++;
                return new LabelledItem(actual.next(), actualClass);
            }
        }

        @Override
        public void remove() {
            throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
        }
    }
}
