package net.snurkabill.neuralnetworks;

public class Tuple {
	private static final double TOLERANCE = 0.0001;
	
	int a;
	int b;
	int num;
	public Tuple(int a, int b) {
		this.a = a;
		this.b = b;
		num = 2 * a + b;
	}
	
	public Tuple(double [] results) {
		this((int)(results[0] + TOLERANCE), (int)(results[1] + TOLERANCE));
	}
}
