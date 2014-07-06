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

    final Map<String, List<MnistItem>> trainingSet = new HashMap<>();
    final Map<String, List<MnistItem>> testSet = new HashMap<>();
	
	final Map<Integer, List<DataItem>> trainingSet2 = new HashMap<>();
	final Map<Integer, List<DataItem>> testSet2 = new HashMap<>();
	

    int rows = 0;
    int cols = 0;
    int count = 0;
    int current = 0;

    public MnistDatasetReader(File labelsFile, File imagesFile) throws IOException, FileNotFoundException {
		try(DataInputStream alabelsBuff = new DataInputStream(new GZIPInputStream(new FileInputStream(labelsFile))); 
			DataInputStream aimagesBuf = new DataInputStream(new GZIPInputStream(new FileInputStream(imagesFile)))) {
		
			labelsBuf = alabelsBuff;
			imagesBuf = aimagesBuf;
			
			verify();

			createTrainingSet();
			
			for (int i = 0; i < testSet.size(); i++) {
				System.out.println(testSet.get(String.valueOf(i)).size());
			}
			
		} finally {
			
		}
    }

    public void createTrainingSet() {
        
		int asdf = 0;
        while (hasMoreElements() /*&& current < 500*/) {
            MnistItem i = nextElement();

            if (asdf % 10 == 0) {
                List<DataItem> l = testSet2.get(Integer.parseInt(i.label));
                if (l == null)
                    l = new ArrayList<DataItem>();
                testSet2.put(Integer.parseInt(i.label), l);

                l.add(new DataItem(i.data));
            } else {
                List<DataItem> l = trainingSet2.get(Integer.parseInt(i.label));
                if (l == null)
                    l = new ArrayList<DataItem>();
                trainingSet2.put(Integer.parseInt(i.label), l);

                l.add(new DataItem(i.data));
            }
			
			asdf++;
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
		
		/*Map<Integer, List<DataItem>> tmp = new HashMap<>();
		
		for (int i = 0; i < trainingSet.size(); i++) {
			tmp.put(i, new ArrayList<DataItem>());
			for (int j = 0; j < trainingSet.get(String.valueOf(i)).size(); j++) {
				MnistItem item = trainingSet.get(String.valueOf(i)).get(j);
				tmp.get(i).add(new DataItem(item.data));
			}
		}*/
		return trainingSet2;
	}
	
	public Map<Integer, List<DataItem>> getTestingData() {
		
		/*Map<Integer, List<DataItem>> tmp = new HashMap<>();
		
		for (int i = 0; i < testSet.size(); i++) {
			tmp.put(i, new ArrayList<DataItem>());
			for (int j = 0; j < testSet.get(String.valueOf(i)).size(); j++) {
				MnistItem item = testSet.get(String.valueOf(i)).get(j);
				tmp.get(i).add(new DataItem(item.data));
			}
		}*/
		return testSet2;
	}
}
