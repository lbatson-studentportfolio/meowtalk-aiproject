public class Neuron {

    // Value represented in the neuron
    private float activation;

    // Value that represented the weighted activation before inputted into the sigmoid function.
    // Used for backpropagation calculus.
    private float weightedSum;

    // Value added to a propagated signal when it reaches this neuron. Input layer neurons are unaffected.
    private float bias;

    // Where 0 is the top of the host layer.
    private final int positionInLayer;

    // Arg constructor 
        // No-arg constructor should not be a thing. 
        // Mis-indexed neurons are not a joke, Jim; millions of neural networks suffer every year.
    public Neuron(float activation, float bias, int positionInLayer, int layerPositionInNetwork) {
        this.activation = activation;
        this.positionInLayer = positionInLayer;
    }

    public float getActivation() {
        return activation;
    }
    
    public float getWeightedSum() {
        return weightedSum;
    }

    public float getBias() {
        return bias;
    }

    public int getPositionInLayer() {
        return positionInLayer;
    }

    public void setBias(float bias) {
        this.bias = bias;
    }

    public void setActivation(float activation) {
        this.activation = activation;
    }

    public void setWeightedSum(float weightedSum) {
        this.weightedSum = weightedSum;
    }

     // Calculates a weighted sum from the activations of the previous layer in reference to the calling neuron.
     // Applies bias of calling neuron to the weighted sum, calculates the sigmoid of the sum, and assigns it to the activation
     // of the calling neuron.
    public void receivePropagationFrom(Neuron[] sourceLayer, float[][] weightMatrix) {

        float weightedSum = 0;

        for (int position = 0; position < sourceLayer.length; position++) {

            // Adds product of previous neuron activation and weight using row (position) and column (calling neuron's position).
            // Then performs the sigmoid of it to more accurately classify data.
            weightedSum += sourceLayer[position].getActivation() * weightMatrix[position][this.getPositionInLayer()];

        }

        // Apply bias.
        weightedSum += this.getBias();

        this.setWeightedSum(weightedSum);
        weightedSum = NeuralNetwork.getSigmoidFrom(weightedSum);

        // Set new weighted sum, completing a forward propagation for a single neuron.
        this.setActivation(weightedSum);

    }

    
}
