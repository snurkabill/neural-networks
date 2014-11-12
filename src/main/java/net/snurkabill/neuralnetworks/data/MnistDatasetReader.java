package net.snurkabill.neuralnetworks.data;

import net.snurkabill.neuralnetworks.data.database.DataItem;
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
    private final DataInputStream labelsBuf;
    private final DataInputStream imagesBuf;
    private final Map<Integer, List<DataItem>> trainingSet;
    private final Map<Integer, List<DataItem>> testSet;
    private int rows = 0;
    private int cols = 0;
    private int count = 0;
    private int current = 0;

    public MnistDatasetReader(File labelsFile, File imagesFile, int sizeOfDataset) throws IOException, FileNotFoundException {
        LOGGER.info("Unzipping files with {} samples ...", sizeOfDataset);
        trainingSet = new LinkedHashMap<>((int) ((double) (sizeOfDataset / TEST_FILTER) * (TEST_FILTER - 1)) + 1, 1);
        testSet = new LinkedHashMap<>((sizeOfDataset / TEST_FILTER) + 1, 1);
        try (DataInputStream alabelsBuff = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile)));
             DataInputStream aimagesBuf = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)))) {
            labelsBuf = alabelsBuff;
            imagesBuf = aimagesBuf;
            verify();
            createTrainingSet(sizeOfDataset);
        }
        System.gc();
    }

    public final void createTrainingSet(int sizeOfDataset) {
        LOGGER.info("Creating hash maps...");
        int elemCounter = 0;
        while (hasMoreElements() && current < sizeOfDataset) {
            MnistItem i = nextElement();
            if (elemCounter % TEST_FILTER == 0) {
                List<DataItem> l = testSet.get(Integer.parseInt(i.label));
                if (l == null)
                    l = new ArrayList<>();
                testSet.put(Integer.parseInt(i.label), l);
                l.add(new DataItem(i.data));
            } else {
                List<DataItem> l = trainingSet.get(Integer.parseInt(i.label));
                if (l == null)
                    l = new ArrayList<>();
                trainingSet.put(Integer.parseInt(i.label), l);
                l.add(new DataItem(i.data));
            }
            elemCounter++;
        }
        LOGGER.info("Maps created!");
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

    public Map<Integer, List<DataItem>> getTrainingData() {
        return trainingSet;
    }

    public Map<Integer, List<DataItem>> getTestingData() {
        return testSet;
    }

}
