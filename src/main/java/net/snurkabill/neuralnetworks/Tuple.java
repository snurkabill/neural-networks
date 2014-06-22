package net.snurkabill.neuralnetworks;

public class Tuple {
	private static final double TOLERANCE = 0.0001;
	
	int a;
	int b;
	int num;
	public Tuple(int a, int b) {
		this.a = a;
		this.b = b;
		calcNum();
	}
	
	public Tuple(double [] results) {
		this.a = (int) (results[0] + TOLERANCE);
		this.b = (int) (results[1] + TOLERANCE);
		calcNum();
	}
	
	private void calcNum() {
		num = 2 * a + b;
	}
}
