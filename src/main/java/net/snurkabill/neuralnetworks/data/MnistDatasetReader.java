package net.snurkabill.neuralnetworks.data;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOError;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.*;
import java.util.zip.GZIPInputStream;

/**
 * Reads the Minst image data from
 */
public class MnistDatasetReader implements Enumeration<MnistItem>
{
    final DataInputStream labelsBuf;
    final DataInputStream imagesBuf;

    SecureRandom r = new SecureRandom();

    final Map<String, List<MnistItem>> trainingSet = new HashMap<String, List<MnistItem>>();
    final Map<String, List<MnistItem>> testSet = new HashMap<String, List<MnistItem>>();

    int rows = 0;
    int cols = 0;
    int count = 0;
    int current = 0;

    public MnistDatasetReader(File labelsFile, File imagesFile)
    {
        try
        {
            labelsBuf = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile)));
            imagesBuf = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)));

            verify();

            createTrainingSet();
        }
        catch (FileNotFoundException e)
        {
            throw new IOError(e);
        }
        catch (IOException e)
        {
            throw new IOError(e);
        }
        finally
        {


        }



    }

    public void createTrainingSet() {
        boolean done = false;

        while (!done || !hasMoreElements()) {
            MnistItem i = nextElement();

            if (r.nextDouble() > 0.3) {
                List<MnistItem> l = testSet.get(i.label);
                if (l == null)
                    l = new ArrayList<MnistItem>();
                testSet.put(i.label, l);

                l.add(i);
            } else {
                List<MnistItem> l = trainingSet.get(i.label);
                if (l == null)
                    l = new ArrayList<MnistItem>();
                trainingSet.put(i.label, l);

                l.add(i);
            }

            if (trainingSet.isEmpty())
                continue;

            boolean isDone = true;
            for (Map.Entry<String, List<MnistItem>> entry : trainingSet.entrySet()) {
                if (entry.getValue().size() < 100) {
                    isDone = false;
                    break;
                }
            }

            done = isDone;
        }
    }

    public MnistItem getTestItem()
    {
        List<MnistItem> list = testSet.get(String.valueOf(r.nextInt(10)));
        return list.get(r.nextInt(list.size()));

    }

    public MnistItem getTrainingItem()
    {
        List<MnistItem> list = trainingSet.get(String.valueOf(r.nextInt(10)));
        return list.get(r.nextInt(list.size()));

    }

    public MnistItem getTrainingItem(int i)
    {
        List<MnistItem> list = trainingSet.get(String.valueOf(i));
        return list.get(r.nextInt(list.size()));

    }

    private void verify() throws IOException
    {
        int magic = labelsBuf.readInt();
        int labelCount = labelsBuf.readInt();

        System.err.println("Labels magic=" + magic + ", count=" + labelCount);

        magic = imagesBuf.readInt();
        int imageCount = imagesBuf.readInt();
        rows = imagesBuf.readInt();
        cols = imagesBuf.readInt();

        System.err.println("Images magic=" + magic + " count=" + imageCount + " rows=" + rows + " cols=" + cols);

        if (labelCount != imageCount)
            throw new IOException("Label Image count mismatch");

        count = imageCount;
    }

    public boolean hasMoreElements()
    {
        return current < count;
    }

    public MnistItem nextElement()
    {
        MnistItem m = new MnistItem();

        try
        {
            m.label = String.valueOf(labelsBuf.readUnsignedByte());
            m.data = new double[rows * cols];

            for (int i = 0; i < m.data.length; i++)
            {
                m.data[i] = imagesBuf.readUnsignedByte();
            }

            return m;
        }
        catch (IOException e)
        {
            current = count;
            throw new IOError(e);
        }
        finally
        {
            current++;
        }

    }
	
	public Map<Integer, List<DataItem>> getTrainingData() {
		
		Map<Integer, List<DataItem>> tmp = new HashMap<>();
		
		for (int i = 0; i < trainingSet.size(); i++) {
			tmp.put(i, new ArrayList<DataItem>());
			for (int j = 0; j < trainingSet.get(String.valueOf(i)).size(); j++) {
				MnistItem item = trainingSet.get(String.valueOf(i)).get(j);
				tmp.get(i).add(new DataItem(item.data));
			}
		}
		return tmp;
	}
	
	public Map<Integer, List<DataItem>> getTestingData() {
		
		Map<Integer, List<DataItem>> tmp = new HashMap<>();
		
		for (int i = 0; i < testSet.size(); i++) {
			tmp.put(i, new ArrayList<DataItem>());
			for (int j = 0; j < testSet.get(String.valueOf(i)).size(); j++) {
				MnistItem item = testSet.get(String.valueOf(i)).get(j);
				tmp.get(i).add(new DataItem(item.data));
			}
		}
		return tmp;
	}
}
