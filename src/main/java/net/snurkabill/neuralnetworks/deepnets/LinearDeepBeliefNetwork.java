package net.snurkabill.neuralnetworks.deepnets;

import net.snurkabill.neuralnetworks.feedforwardnetwork.FeedForwardableNetwork;

public class LinearDeepBeliefNetwork extends FeedForwardableNetwork {

    private final StuckedRBM rbms;

    public LinearDeepBeliefNetwork(String name, StuckedRBM rbm) {
        super(name, "LinearDBN");
        this.rbms = rbm;
    }
	
	/**
	 * TODO : everything .... (after RBM chimney put linear classificators FOR MORE FUN!)
	 */

}
