package net.snurkabill.neuralnetworks.data.mnist;

import net.snurkabill.neuralnetworks.newdata.database.NewDataItem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * Reads the Minst image data from
 */
public class MnistDatasetReader implements Enumeration<MnistItem> {

    private static final Logger LOGGER = LoggerFactory.getLogger("MNISTReader");
    private static final int TEST_FILTER = 10;
    private static final int VALIDATION_FILTER = 9;
    private static final int NUM_OF_CLASSEA = 10;
    private final DataInputStream labelsBuf;
    private final DataInputStream imagesBuf;
    private final Map<Integer, List<NewDataItem>> trainingSet;
    private final Map<Integer, List<NewDataItem>> testSet;
    private final Map<Integer, List<NewDataItem>> validationSet;
    private int rows = 0;
    private int cols = 0;
    private int count = 0;
    private int current = 0;


    public MnistDatasetReader(File labelsFile, File imagesFile, int sizeOfDataset, boolean binary) throws IOException {
        LOGGER.info("Unzipping files with {} samples ...", sizeOfDataset);
        trainingSet = new LinkedHashMap<>((int) ((double) (sizeOfDataset / TEST_FILTER) * (TEST_FILTER - 1)) + 1, 1);
        testSet = new LinkedHashMap<>((sizeOfDataset / TEST_FILTER) + 1, 1);
        validationSet = new LinkedHashMap<>();
        try (DataInputStream alabelsBuff = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile)));
             DataInputStream aimagesBuf = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)))) {
            labelsBuf = alabelsBuff;
            imagesBuf = aimagesBuf;
            verify();
            createTrainingSet(sizeOfDataset, binary);
        }
        System.gc();
    }

    public final void createTrainingSet(int sizeOfDataset, boolean binary) {
        LOGGER.info("Loading data...");
        int elemCounter = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            trainingSet.put(i, new ArrayList<>());
            validationSet.put(i, new ArrayList<>());
            testSet.put(i, new ArrayList<>());
        }
        while (hasMoreElements() && current < sizeOfDataset) {
            MnistItem i = nextElement();
            if (binary) {
                for (int j = 0; j < i.data.length; j++) {
                    i.data[j] = (i.data[j] < 1 ? 0 : 1);
                }
            }
            int _class = Integer.parseInt(i.label);
            double[] target = new double[NUM_OF_CLASSEA];
            target[_class] = 1;
            if (elemCounter % TEST_FILTER == 0) {
                testSet.get(_class).add(new NewDataItem(i.data));
            } else {
                if (elemCounter % VALIDATION_FILTER == 0) {
                    validationSet.get(_class).add(new NewDataItem(i.data));
                } else {
                    trainingSet.get(_class).add(new NewDataItem(i.data));
                }
            }
            elemCounter++;
        }
        LOGGER.info("Data loaded!");
    }

    private void verify() throws IOException {
        LOGGER.info("Verifying");
        int magic = labelsBuf.readInt();
        int labelCount = labelsBuf.readInt();
        LOGGER.info("Labels magic = {}, count = {}", magic, labelCount);
        magic = imagesBuf.readInt();
        int imageCount = imagesBuf.readInt();
        rows = imagesBuf.readInt();
        cols = imagesBuf.readInt();
        LOGGER.info("Images magic = {} count = {} rows = {} cols = {}", magic, labelCount, rows, cols);
        if (labelCount != imageCount)
            throw new IOException("Label Image count mismatch");
        count = imageCount;
        LOGGER.info("Verified!");
    }

    @Override
    public boolean hasMoreElements() {
        return current < count;
    }

    @Override
    public MnistItem nextElement() {
        MnistItem item = new MnistItem();
        try {
            item.label = String.valueOf(labelsBuf.readUnsignedByte());
            item.data = new double[rows * cols];
            for (int i = 0; i < item.data.length; i++) {
                item.data[i] = imagesBuf.readUnsignedByte();
            }
            return item;
        } catch (IOException e) {
            current = count;
            throw new IOError(e);
        } finally {
            current++;
        }
    }

    public NewDataItem[][] getTrainingData() {
        NewDataItem[][] items = new NewDataItem[NUM_OF_CLASSEA][];
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            items[i] = new NewDataItem[trainingSet.get(i).size()];
        }
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < trainingSet.get(i).size(); j++) {
                double[] classes = new double[NUM_OF_CLASSEA];
                classes[i] = 1;
                items[i][j] = trainingSet.get(i).get(j);
                items[i][j].target = classes;
            }
        }
        return items;
    }

    public NewDataItem[][] getValidationData() {
        NewDataItem[][] items = new NewDataItem[NUM_OF_CLASSEA][];
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            items[i] = new NewDataItem[validationSet.get(i).size()];
        }
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < validationSet.get(i).size(); j++) {
                double[] classes = new double[NUM_OF_CLASSEA];
                classes[i] = 1;
                items[i][j] = validationSet.get(i).get(j);
                items[i][j].target = classes;
            }
        }
        return items;
    }

    public NewDataItem[][] getTestingData() {
        NewDataItem[][] items = new NewDataItem[NUM_OF_CLASSEA][];
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            items[i] = new NewDataItem[testSet.get(i).size()];
        }
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < testSet.get(i).size(); j++) {
                double[] classes = new double[NUM_OF_CLASSEA];
                classes[i] = 1;
                items[i][j] = testSet.get(i).get(j);
                items[i][j].target = classes;
            }
        }
        return items;
    }

    public NewDataItem[] getRegressionTrainingData() {
        int size = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            size += trainingSet.get(i).size();
        }
        NewDataItem[] items = new NewDataItem[size];
        int index = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < trainingSet.get(i).size(); j++, index++) {
                items[index] = trainingSet.get(i).get(j);
            }
        }
        return items;
    }

    public NewDataItem[] getRegressionValidationData() {
        int size = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            size += validationSet.get(i).size();
        }
        NewDataItem[] items = new NewDataItem[size];
        int index = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < validationSet.get(i).size(); j++, index++) {
                items[index] = validationSet.get(i).get(j);
            }
        }
        return items;
    }

    public NewDataItem[] getRegressionTestingData() {
        int size = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            size += testSet.get(i).size();
        }
        NewDataItem[] items = new NewDataItem[size];
        int index = 0;
        for (int i = 0; i < NUM_OF_CLASSEA; i++) {
            for (int j = 0; j < testSet.get(i).size(); j++, index++) {
                items[index] = testSet.get(i).get(j);
            }
        }
        return items;
    }

}
