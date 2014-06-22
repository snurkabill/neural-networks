package net.snurkabill.neuralnetworks;

import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.security.SecureRandom;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Level;
import java.util.logging.Logger;

public class VolatilityDataReader {

	private final SecureRandom r = new SecureRandom();
	
	private final int numberOfBufferedVolatilities = 5;
	private final int sizeOfBufferedVector = 6;
	private final int additionalSize = 3; // day, hour, minute
	
	private final int wholeSizeOfInputVector = numberOfBufferedVolatilities * sizeOfBufferedVector + additionalSize;
	
	private final Map<Integer, List<VolatilityItem>> trainingSet;
	private final Map<Integer, List<VolatilityItem>> testingSet;
	
	public VolatilityDataReader(List<File> file) {
		trainingSet = new HashMap<Integer, List<VolatilityItem>>();
		testingSet = new HashMap<Integer, List<VolatilityItem>>();
		
		for (int i = 0; i < file.size(); i++) {
			trainingSet.put(i, new ArrayList<VolatilityItem>());
			testingSet.put(i, new ArrayList<VolatilityItem>());
		}
		
		byte[] buffer = new byte[4];
		for (int i = 0; i < file.size(); i++) {
			try {
				DataInputStream tmp = new DataInputStream(new FileInputStream(new File("test_file_BUFFERED_" + i)));
				
				for (int j = 0; tmp.available() != 0; j++) {
					
					int[] inputVector = new int[wholeSizeOfInputVector];
					for (int k = 0; k < inputVector.length; k++) {
						tmp.read(buffer);
						inputVector[k] = (buffer[0] & 0xFF) | (buffer[1] & 0xFF) << 8 | (buffer[2] & 0xFF) << 16 | (buffer[3] & 0xFF) << 24;;
					}
					
					VolatilityItem item = new VolatilityItem();
					
					item.data = inputVector;
					item.label = i;
					
					/*for (int k = 0; k < inputVector.length; k++) {
						System.out.println(inputVector[k]);
					}
					System.out.println("-------------------------------------------");
					*/
					//System.out.println(i + "   " + j);
					if(j % 10 == 0 ) {
						testingSet.get(i % 10).add(item);
					} else {
						trainingSet.get(i % 10).add(item);
					}
				}
			} catch (FileNotFoundException ex) {
				Logger.getLogger(VolatilityDataReader.class.getName()).log(Level.SEVERE, "WELL, FUCK!", ex);
			} catch (IOException ex) {
				Logger.getLogger(VolatilityDataReader.class.getName()).log(Level.SEVERE, "WELL, FUCK2!", ex);
			}	
		}
	}
	
	public VolatilityItem getRandomTestingSample() {
		return getRandomTestingSample(r.nextInt(testingSet.size()));
	}
	
	public VolatilityItem getRandomTrainingSample() {
		return getRandomTrainingSample(r.nextInt(trainingSet.size()));
	}

	public VolatilityItem getRandomTestingSample(int i) {
		return testingSet.get(i).get(r.nextInt(testingSet.get(i).size()));
	}
	
	public VolatilityItem getRandomTrainingSample(int i) {
		return trainingSet.get(i).get(r.nextInt(trainingSet.get(i).size()));
	}

	public int getWholeSizeOfInputVector() {
		return wholeSizeOfInputVector;
	}
	
}
