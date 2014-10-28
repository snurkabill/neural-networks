package net.snurkabill.neuralnetworks.pcatransform;

/**
 * A set of convenience assert-like methods that throw an exception if
 * given condition is not met.
 */
public class Assume {
	public static void assume(boolean expression){
		assume(expression, "");
	}

	public static void assume(boolean expression, String comment){
		if(!expression){
			throw new RuntimeException(comment);
		}
	}
}
