package net.snurkabill.neuralnetworks.pcatransform.covmatrixevd;

import Jama.Matrix;

public class EVDResult {
	public Matrix d;
	public Matrix v;
	
	public EVDResult(Matrix d, Matrix v) {
		this.d = d;
		this.v = v;
	}
}
