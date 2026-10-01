// This is a static class, because initializing neural network(s) with 3 million parameters would end my life.

 import javax.sound.sampled.*;
 import java.io.File;
 import java.io.PrintWriter;
 import java.io.FileNotFoundException;
 import java.io.IOException;
 import java.util.Scanner;
 
 public class NeuralNetwork {
 
     // Input layer
     private static Neuron[] inputLayer = new Neuron[12000];
 
     // First hidden layer
     private static Neuron[] firstHiddenLayer = new Neuron[240];
 
     // Second hidden layer
     private static Neuron[] secondHiddenLayer = new Neuron[240];
 
     // Output layer
     private static Neuron[] outputLayer = new Neuron[5]; // Length is equal to the amount of "classes" of output.
     // (e.g. content, down, agitated, demanding, curious).
 
     // Weights of neuron connections presented as a matrix of floats, use matrix-vector multiplication to propagate signals.
     // Elements within the matrices will be assigned a proper value during the loading phase of the program.
     private static float[][] inputToFirstHiddenWeightMatrix = new float[inputLayer.length][firstHiddenLayer.length];
     private static float[][] firstHiddenToSecondHiddenWeightMatrix = new float[firstHiddenLayer.length][secondHiddenLayer.length];
     private static float[][] secondHiddenToOutputWeightMatrix = new float[secondHiddenLayer.length][outputLayer.length];
 
     private NeuralNetwork() {} // NeuralNetwork is a static class; constructors aren't needed.
 
     // Accessor methods
     // Yes, I know they pass the reference, defeats the purpose of private variables, whatever.
     // The Neuron class depends on this, but I might change the retrieval of the layers/matrices in some way.
     public static float[][] getInputToFirstHiddenWeights() {
         return inputToFirstHiddenWeightMatrix;
     }
     public static float[][] getFirstHiddenToSecondHiddenWeights() {
         return firstHiddenToSecondHiddenWeightMatrix;
     }
     public static float[][] getSecondHiddenToOutputWeights() {
         return secondHiddenToOutputWeightMatrix;
     }
     public static Neuron[] getInputLayer() {
         return inputLayer;
     }
     public static Neuron[] getFirstHiddenLayer() {
         return firstHiddenLayer;
     }
     public static Neuron[] getSecondHiddenLayer() {
         return secondHiddenLayer;
     }
     public static Neuron[] getOutputLayer() {
         return outputLayer;
     }
 
    // Method used to encapsulate the workings behind training the algorithm so that dividing the processes when debugging
    // will not be a huge pain.
    public static void trainNetwork(String batchName) {

        // Instantiate matrices/vectors to hold the shifts in the weights and biases.
        // Each shift is added to the relevant value in the relevant array.
        float[][] shiftsInInputToFirstHiddenWeightMatrix = new float[inputLayer.length][firstHiddenLayer.length];
        float[][] shiftsInFirstHiddenToSecondHiddenWeightMatrix = new float[firstHiddenLayer.length][secondHiddenLayer.length];
        float[][] shiftsInSecondHiddenToOutputWeightMatrix = new float[secondHiddenLayer.length][outputLayer.length];
        float[] shiftsInFirstHiddenLayerBiasVector = new float[firstHiddenLayer.length];
        float[] shiftsInSecondHiddenLayerBiasVector = new float[secondHiddenLayer.length];
        float[] shiftsInOutputLayerBiasVector = new float[outputLayer.length];

        // Load batch(es) from audio training data in the project files directory.
        File batchDirectory = new File("trainingdatafiles\\" + batchName);
        File[] trainingExamples = TrainingData.getBatchFrom(batchDirectory);
        TrainingData[] batch = new TrainingData[trainingExamples.length];

        // Loop through the training examples, create a new batch with TrainingData objects.
        for (int index = 0; index < trainingExamples.length; index++) {
            float[] expectedOutput = new float[5];

            // Create expected output dependent on the first number in the name of a file.
            // Keeps all other values at 0.0 (0%), but makes the expected output 1.0 (100%).
            int activeNeuronPosition = Character.getNumericValue(trainingExamples[index].getName().charAt(0));
            expectedOutput[activeNeuronPosition] = (float) 1.0;

            batch[index] = new TrainingData(trainingExamples[index], expectedOutput);

            // Debug
            // System.out.println(batch[index].getAudioFile().getName() + ":"
            //                     + "\n\t" + batch[index].getExpectedOutputAt(0)
            //                     + "\n\t" + batch[index].getExpectedOutputAt(1)
            //                     + "\n\t" + batch[index].getExpectedOutputAt(2)
            //                     + "\n\t" + batch[index].getExpectedOutputAt(3)
            //                     + "\n\t" + batch[index].getExpectedOutputAt(4));
        }

        // Begin main loop.
        for (TrainingData trainingData : batch) {

            System.out.println("Running training batch");

            // Holds numerical representation of the current layer's position in the network
            // Facilitates backpropagation calculus.
            int layerPositionInNetwork;

            // Clear the network to prevent inaccurate propagations.
            NeuralNetwork.clearNetwork();

            // Run the file through the network, and calculate the cost.
            try {

                prepareInputLayer(trainingData.getAudioFile());

                // Begin propagation of signals. Let's get this show on the road.
                // Input layer -> first hidden layer propagation
                NeuralNetwork.forwardPropagate(
                    NeuralNetwork.getInputLayer(), 
                    NeuralNetwork.getFirstHiddenLayer(), 
                    NeuralNetwork.getInputToFirstHiddenWeights()
                );
                
                // First hidden layer -> second hidden layer propagation
                NeuralNetwork.forwardPropagate(
                    NeuralNetwork.getFirstHiddenLayer(), 
                    NeuralNetwork.getSecondHiddenLayer(), 
                    NeuralNetwork.getFirstHiddenToSecondHiddenWeights()
                );

                // Second hidden layer -> output layer propagation
                NeuralNetwork.forwardPropagate(
                    NeuralNetwork.getSecondHiddenLayer(), 
                    NeuralNetwork.getOutputLayer(), 
                    NeuralNetwork.getSecondHiddenToOutputWeights()
                );

                // Iterate through each network layer backwards, add nudges to matrices and vectors.
                layerPositionInNetwork = 3; // Starts at last layer (output layer).
                addNudgesToGradients(
                    trainingData, // current training example.
                    shiftsInSecondHiddenToOutputWeightMatrix, // shifts in weights matrix
                    shiftsInOutputLayerBiasVector, // shifts in biases vector
                    outputLayer, // current layer
                    secondHiddenLayer, // previous layer
                    layerPositionInNetwork
                );

                System.out.println("Added shifts to secondhiddentooutputweightmatrix");
                System.out.println("Added shifts to outputlayerbiasvector");

                // Backpropagate. (HE SAID THE THING!!!)
                layerPositionInNetwork = 2;
                addNudgesToGradients(
                    trainingData,
                    shiftsInFirstHiddenToSecondHiddenWeightMatrix,
                    shiftsInSecondHiddenLayerBiasVector,
                    secondHiddenLayer,
                    firstHiddenLayer,
                    layerPositionInNetwork
                );

                System.out.println("Added shifts to firsthiddentosecondhiddenweightmatrix");
                System.out.println("Added shifts to secondhiddenlayerbiasvector");

                // Backpropagate for the final time.
                layerPositionInNetwork = 1;
                addNudgesToGradients(
                    trainingData,
                    shiftsInInputToFirstHiddenWeightMatrix,
                    shiftsInFirstHiddenLayerBiasVector,
                    firstHiddenLayer,
                    inputLayer,
                    layerPositionInNetwork
                );

                System.out.println("Added shifts to inputtofirsthiddenweightmatrix");
                System.out.println("Added shifts to firsthiddenlayerbiasvector");

                // Debug
                // Calculate cost of current training example.
                float cost = calculateCost(trainingData);
                System.out.println("Cost: " + cost);

            }
            catch (Exception exception) {
                System.out.println("One of the training data in the batch failed to process. Moving to next.");
            }

        }

        // All the shifts are computed, now it's time to substract the values in the original parameter matrices/vectors
        // by the average of the shifts in the relevant gradients.
        // The reason why I'm substracting the values is because these are negative gradients.
        // This is disgustingly computationally inefficient, but at this late hour in development, I might as well be
        // crapping out code.

        // Shifts in input layer to first hidden layer weight matrix.
        for (int row = 0; row < shiftsInInputToFirstHiddenWeightMatrix.length; row++) {
            for (int column = 0; column < shiftsInInputToFirstHiddenWeightMatrix[row].length; column++) {
                inputToFirstHiddenWeightMatrix[row][column] -= shiftsInInputToFirstHiddenWeightMatrix[row][column] /= batch.length;
            }
        }

        // Shifts in first hidden layer to second hidden layer weight matrix.
        for (int row = 0; row < shiftsInFirstHiddenToSecondHiddenWeightMatrix.length; row++) {
            for (int column = 0; column < shiftsInFirstHiddenToSecondHiddenWeightMatrix[row].length; column++) {
                firstHiddenToSecondHiddenWeightMatrix[row][column] -= shiftsInFirstHiddenToSecondHiddenWeightMatrix[row][column] / batch.length;
            }
        }

        // Shifts in second hidden layer to output layer weight matrix.
        for (int row = 0; row < shiftsInSecondHiddenToOutputWeightMatrix.length; row++) {
            for (int column = 0; column < shiftsInSecondHiddenToOutputWeightMatrix[row].length; column++) {
                secondHiddenToOutputWeightMatrix[row][column] -= shiftsInSecondHiddenToOutputWeightMatrix[row][column] / batch.length;
            }
        }

        // Shifts in first hidden layer bias vector.
        for (int index = 0; index < shiftsInFirstHiddenLayerBiasVector.length; index++) {
            firstHiddenLayer[index].setBias(firstHiddenLayer[index].getBias() - shiftsInFirstHiddenLayerBiasVector[index] / batch.length);
        }

        // Shifts in second hidden layer bias vector.
        for (int index = 0; index < shiftsInSecondHiddenLayerBiasVector.length; index++) {
            secondHiddenLayer[index].setBias(secondHiddenLayer[index].getBias() - shiftsInSecondHiddenLayerBiasVector[index] / batch.length);
        }

        // Shifts in output layer bias vector.
        for (int index = 0; index < shiftsInOutputLayerBiasVector.length; index++) {
            outputLayer[index].setBias(outputLayer[index].getBias() - shiftsInOutputLayerBiasVector[index] / batch.length);
        }

        // Write the new changes to the relevant text files in the project files.
        File file;
        
        // InputToFirstHiddenWeights
        file = new File("InputToFirstHiddenWeights.txt");
        writeToFile(file, inputToFirstHiddenWeightMatrix);
        file = null;

        // FirstHiddenToSecondHiddenWeights
        file = new File("FirstHiddenToSecondHiddenWeights.txt");
        writeToFile(file, firstHiddenToSecondHiddenWeightMatrix);
        file = null;

        // SecondHiddenToOutputWeights
        file = new File("SecondHiddenToOutputWeights.txt");
        writeToFile(file, secondHiddenToOutputWeightMatrix);
        file = null;

        // FirstHiddenLayerBiases
        file = new File("FirstHiddenLayerBiases.txt");
        writeToFile(file, firstHiddenLayer);
        file = null;

        // SecondHiddenLayerBiases
        file = new File("SecondHiddenLayerBiases.txt");
        writeToFile(file, secondHiddenLayer);
        file = null;

        // OutputLayerBiases
        file = new File("OutputLayerBiases.txt");
        writeToFile(file, outputLayer);
        file = null;

        // It's done.
        System.out.println("Successfully trained the neural network using a mini-batch of " + batch.length + " training examples.");

    }

    // Writes the values of the source weight matrix in the program to the relevant text file on the disk.
    // Meant to be called after the backpropagation calculus and training is performed.
    public static void writeToFile(File originalMatrixFile, float[][] sourceMatrix) {

        PrintWriter printWriter = null;

        try {
            printWriter = new PrintWriter(originalMatrixFile);
        }
        catch (FileNotFoundException exception) {
            exception.printStackTrace();
        }

        for (int row = 0; row < sourceMatrix.length; row++) {
            for (int column = 0; column < sourceMatrix[row].length; column++) {
                printWriter.println(sourceMatrix[row][column]);
            }
        }

        printWriter.close();

    }

    // Writes the values of the source bias vector (actually neurons) in the program to the relevant text file on the disk.
    // Meant to be called after the backpropagation calculus and training is performed.
    // Utilizes neurons instead of a new bias vector.
    public static void writeToFile(File originalVectorFile, Neuron[] sourceLayer) {

        PrintWriter printWriter = null;

        try {
            printWriter = new PrintWriter(originalVectorFile);
        }
        catch (FileNotFoundException exception) {
            exception.printStackTrace();
        }

        for (int index = 0; index < sourceLayer.length; index++) {
            printWriter.println(sourceLayer[index].getBias());
        }

        printWriter.close();

    }

    // Calculates the partial derivative of the weights & biases in a specified layer.
    public static void addNudgesToGradients(
        TrainingData trainingData,
        float[][] specifiedWeightGradient, 
        float[] specifiedBiasGradient, 
        Neuron[] currentLayer, 
        Neuron[] previousLayer,
        int layerPositionInNetwork) {

        /*
         * NOTES
         * 
         * 1. specifiedWeightGradient references the shifts in a weight matrix defined in trainNetwork()
         *      Any values added to specifiedWeightGradient changes the reference.
         * 2. specifiedBiasGradient references the shfits in a bias vector defined in trainNetwork()
         *      Any values added to specifiedBiasGradient changes the reference.
         * 3. The current and previous layers equal the passed references of actual layers in the network.
         *      THESE SHOULD NOT CHANGE NO MATTER THE CIRCUMSTANCE.
         */

        // Weight gradient.
        // For each neuron in the previous array:
        for (int k = 0; k < previousLayer.length; k++) {

            // For each neuron in the current array:
            for (int j = 0; j < currentLayer.length; j++) {

                // Partial derivative of cost function w/ respect to partial derivative of weight at indices j & k.
                float delCdelW = (float) 0.0;

                // Partial derivative of weighted sum at index j w/ respect to partial derivative of weight at indices j & k.
                float delZdelW = previousLayer[k].getActivation();
                // Partial derivative of neuron activation at index j w/ respect to partial derivative of weighted sum at index j.
                float delAdelZ = getSigmoidDerivativeFrom(currentLayer[j].getWeightedSum());
                System.out.println(delAdelZ);
                // Partial derivative of cost function w/ respect to partial derivative of neuron activation at index j.
                float delCdelA = calculatePDCostOfActivationInLayer(trainingData, layerPositionInNetwork, j);

                // Chain rule of partial derivatives, calculate partial derivative of cost function w/ respect to weight
                delCdelW = delZdelW * delAdelZ * delCdelA;

                // Adds partial derivative to the weight gradient.
                specifiedWeightGradient[k][j] += delCdelW;

            }

        }

        // Bias gradient.
        for (int j = 0; j < currentLayer.length; j++) {

            // Partial derivative of cost function w/ respect to partial derivative of bias at index j.
            float delCdelB = (float) 0.0;

            // Partial derivative of weighted sum at index j w/ respect to partial derivative of bias at index j.
            // OMITTED BECAUSE DELZDELB EQUALS 1 DUE TO POWER RULE.

            // Partial derivative of neuron activation at index j w/ respect to partial derivative of weighted sum at index j.
            float delAdelZ = getSigmoidDerivativeFrom(currentLayer[j].getWeightedSum());

            // Partial derivative of the cost function w/ respect to partial derivative of neuron activation at index j.
            float delCdelA = calculatePDCostOfActivationInLayer(trainingData, layerPositionInNetwork, j);

            // Chain rule of partial derivatives, calculate partial derivative of cost function w/ respect to bias.
            delCdelB = delAdelZ * delCdelA;

            // Add partial derivative to the bias gradient.
            specifiedBiasGradient[j] += delCdelB;

        }

    }

    // Calculates the partial derivative of the cost function w/ respect to the activation of a neuron in some layer L <= current layer.
    // This function is recursive (just kill me).
    public static float calculatePDCostOfActivationInLayer(TrainingData trainingData, int layerPosition, int neuronPosition) {

        // Used for recursion when calculating the partial derivative.
        // In some cases, the program has to recursively calculate forward until the output layer to find how an activation
        // affects the cost function.
        Neuron[] nextLayer = null;
        float[][] weightMatrix = null;

        // Holds return value of recursive calls.
        float delCdelA = (float) 0.0;

        // If layer position represents the output layer
        if (layerPosition == 3) {

            delCdelA = (float) (2.0 * (outputLayer[neuronPosition].getActivation() - trainingData.getExpectedOutputAt(neuronPosition)));

        }
        else {

            if (layerPosition == 2) {
                nextLayer = outputLayer;
                weightMatrix = secondHiddenToOutputWeightMatrix;
            }
            else if (layerPosition == 1) {
                nextLayer = secondHiddenLayer;
                weightMatrix = firstHiddenToSecondHiddenWeightMatrix;
            }

            // Summation of partial derivatives of the cost function w/ respect to the activation in the previous layer.
            for (int nextLayerNeuronPosition = 0; nextLayerNeuronPosition < nextLayer.length; nextLayerNeuronPosition++) {

                // Add result of each iteration to the partial derivative.
                delCdelA += (
                    weightMatrix[neuronPosition][nextLayerNeuronPosition]
                    * getSigmoidDerivativeFrom(nextLayer[nextLayerNeuronPosition].getWeightedSum())
                    * calculatePDCostOfActivationInLayer(trainingData, layerPosition + 1, nextLayerNeuronPosition)
                );

            }

        }

        return delCdelA;

    }

      // Propagates the signals of neurons in a layer to every neuron in the next.
      // Looped version of Neuron's instance method receivePropagationFrom(params).
     public static void forwardPropagate(Neuron[] sourceLayer, Neuron[] targetLayer, float[][] weightMatrix) {
 
        for (int position = 0; position < targetLayer.length; position++) {
 
             Neuron receivingNeuron = targetLayer[position];
             receivingNeuron.receivePropagationFrom(sourceLayer, weightMatrix);
 
        }
 
     }
 
     // Calculate and return the cost function with respect to a specific training example.
     // This is used for back propagation to shift weights/biases in the neural network.
     public static float calculateCost(TrainingData trainingData) {
        
        float cost = 0;

        for (int neuronPosition = 0; neuronPosition < outputLayer.length; neuronPosition++) {

            // Actual output at neuronPosition minus the expected output at neuronPosition is squared and added to cost.
            cost += Math.pow( outputLayer[neuronPosition].getActivation() - trainingData.getExpectedOutputAt(neuronPosition), 2 );

        }

        return cost;

     }
 
      // Calculates the output of a sigmoid function.
      // Returns a continuous float value from 0.0 to 1.0.
     public static float getSigmoidFrom(float x) {
 
         float result = 0;
 
         // Sigmoid function
         result = (float) (1.0 / (1 + Math.exp(-(x))));
 
         // Return a value from 0.0 to 1.0.
         return result;
 
     }

     // Derivative of sigmoid function above.
     public static float getSigmoidDerivativeFrom(float x) {

        float result = 0;

        // Derivative of sigmoid
        result = (float) ( getSigmoidFrom(x) * ( 1 - getSigmoidFrom(x) ) );

        return result;

     }
 
      // Takes all data from output classes (data classes, not java classes), then makes an assumption of what the cat may be
      // expressing based on the probability represented by the output neurons.
     public static void printCalculatedExpression() {
 
         // Classes of data represented by output layer of neural network.
         float contentment = (float) 0.0;
         float downness = (float) 0.0;
         float agitation = (float) 0.0;
         float demanding = (float) 0.0;
         float curiosity = (float) 0.0;
 
         contentment = outputLayer[0].getActivation();
         downness = outputLayer[1].getActivation();
         agitation = outputLayer[2].getActivation();
         demanding = outputLayer[3].getActivation();
         curiosity = outputLayer[4].getActivation();
 
         float[] traits = {contentment, downness, agitation, demanding, curiosity};
         
         // A formatted print statement makes this WAAAY easier.
         System.out.printf("The program has detected these traits from your cat's speech: "
                                 + "\nContent: %.2f%%"
                                 + "\nDown: %.2f%%"
                                 + "\nAgitated: %.2f%%"
                                 + "\nDemanding: %.2f%%"
                                 + "\nCurious: %.2f%%\n",
                                 contentment * 100, downness * 100, agitation * 100, demanding * 100, curiosity * 100);
                                 // Traits are multiplied by 100 to get a percentage value.
 
         // Determine what the cat is most likely trying to express.
         String[] result = getTwoMostProbableEmotions(traits);

         System.out.println("Your cat seems to be mostly "
                            + result[0]
                            + " and somewhat "
                            + result[1]);

     }
 
     // Sequentially fill the input layer using an audioinputstream from an audio file.
     public static void prepareInputLayer(File file) throws IOException, UnsupportedAudioFileException {
 
         // Initialize an AudioInputStream object with the file to be read from.
         AudioInputStream audioInputStream = AudioSystem.getAudioInputStream(file);
 
         // Initialize values used for quantifying data.
         int neuronPosition = 0;
         int bytesInFrame = audioInputStream.getFormat().getFrameSize();
         byte[] audioDataBuffer = new byte[1024 * bytesInFrame];
             // 1024 is a number chosen to support large enough buffers to improve data literacy while maintaining available memory.
             // Multiply that by 2, the number of bytes in a frame, to make sure we read a whole frame.
             // The number itself is arbitrarily chosen somewhat, but recommended by the java audio programming documentation.
 
         // Begin reading data from the audio stream. General checked exceptions are caught.
         try {
 
             // Variables used to quantify and analyze the amount of data moving through the system.
             short newSum = 0; // Holds two bytes of data. That's extremely convenient.
             boolean endOfStream = false;
             
             // If the end of the audio stream has NOT been reached...
             while (audioInputStream.read(audioDataBuffer, 0, audioDataBuffer.length) != -1) {
 
                 // Loop through the buffer and apply byte to a neuron in the input layer.
                 for (int index = 0; index < audioDataBuffer.length; index += 2) {
                     
                     // Assign newSum the bytes from the buffer.
                     // If the second byte is 0, only use the first byte's value.
                     newSum = (audioDataBuffer[index + 1] != 0) ?
                        (short) (audioDataBuffer[index] * audioDataBuffer[index + 1])
                        : (short) (audioDataBuffer[index]);
 
                     // Assign neuron's value to the new sum.
                     // If the target neuron is out of bounds of the input layer, end the stream prematurely.
                     try {
                        // Apply a unique activation function, following the sigmoid but adding a coefficient to the input
                        // value, that represents 1/45, the reciprocal of the max decibels the system should support.
                         // float newSigmoid = (float) getSigmoidFrom(newSum, 45);
                         // System.out.println(newSigmoid);

                         // Assign new sigmoid activation.
                         inputLayer[neuronPosition].setActivation(newSum);
                     }
                     catch (ArrayIndexOutOfBoundsException exception) {
                         endOfStream = true;
                         break;
                     }
 
                     // Increment neuron position.
                     neuronPosition++;
 
                 }
 
                 //If the stream ends prematurely, i.e. both bytes in an iteration are zero, end the processing of data.
                 if (endOfStream) {
                     break;
                 }
 
            }
 
         }
         catch (Exception exception) {
             System.out.println(exception);
         }
 
     }

     // Get two most probable emotions and return it as a string array of two elements.
     public static String[] getTwoMostProbableEmotions(float... values) {

        String[] traitNames = {"content", "down", "agitated", "demanding", "curious"};
        float greatest = 0;
        float secondGreatest = 0;
        int greatestIndex = 0;
        int secondGreatestIndex = 0;
        String[] result = new String[2];

        for (int index = 0; index < values.length; index++) {

            if (values[index] > greatest) {
                secondGreatest = greatest;
                secondGreatestIndex = greatestIndex;
                greatest = values[index];
                greatestIndex = index;
            }
            else if (values[index] > secondGreatest) {
                secondGreatest = values[index];
                secondGreatestIndex = index;
            }

        }

        result[0] = traitNames[greatestIndex];
        result[1] = traitNames[secondGreatestIndex];

        return result;
 
     }

    // Loads the parameters, weights and biases, by taking values from the text files and importing them into neurons and matrices.
     public static void loadParameters() throws FileNotFoundException {
 
         // Use separate methods here
         loadInputLayer();
         loadFirstHiddenBiases();
         loadSecondHiddenBiases();
         loadOutputBiases();
         loadInputToFirstHiddenWeightMatrix();
         loadFirstHiddenToSecondHiddenWeightMatrix();
         loadSecondHiddenToOutputWeightMatrix();
 
     }

     // Construct neurons for input layer.
     public static void loadInputLayer() {
        
        for (int index = 0; index < inputLayer.length; index++) {

            inputLayer[index] = new Neuron((float) 0, (float) 0, index, 0);

        }

     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads biases into the hidden layer using the text file.
      */
     public static void loadFirstHiddenBiases() throws FileNotFoundException {
 
         // Prepare biases for hidden layer.
         File hiddenLayerBiases = new File("FirstHiddenLayerBiases.txt");
         int neuronPosition = 0;
         Scanner fileScanner = new Scanner(hiddenLayerBiases);
 
         while (fileScanner.hasNextFloat()) {
             // Construct a new neuron with sum 0.0, the bias from the file, and the neuron's position.
             firstHiddenLayer[neuronPosition] = new Neuron((float) 0.0, (float) fileScanner.nextFloat(), neuronPosition, 1);
             neuronPosition++;
         }
 
         // Free up space.
         hiddenLayerBiases = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads biases into the second hidden layer using the text file.
      */
     public static void loadSecondHiddenBiases() throws FileNotFoundException {
 
         // Prepare biases for hidden layer.
         File secondHiddenLayerBiases = new File("SecondHiddenLayerBiases.txt");
         int neuronPosition = 0;
         Scanner fileScanner = new Scanner(secondHiddenLayerBiases);
 
         while (fileScanner.hasNextFloat()) {
             // Construct a new neuron with sum 0.0, the bias from the file, and the neuron's position.
             secondHiddenLayer[neuronPosition] = new Neuron((float) 0.0, (float) fileScanner.nextFloat(), neuronPosition, 2);
             neuronPosition++;
         }
 
         // Free up space.
         secondHiddenLayerBiases = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads biases into the output layer using the text file.
      */
     public static void loadOutputBiases() throws FileNotFoundException {
 
         // Prepare biases for output layer.
         File outputLayerBiases = new File("OutputLayerBiases.txt");
         int neuronPosition = 0;
         Scanner fileScanner = new Scanner(outputLayerBiases);
 
         while (fileScanner.hasNextFloat()) {
             // Construct a new neuron with sum 0.0, the bias from the file, and the neuron's position.
             outputLayer[neuronPosition] = new Neuron((float) 0.0, (float) fileScanner.nextFloat(), neuronPosition, 3);
             neuronPosition++;
         }
 
         // Free up space.
         outputLayerBiases = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads weights into the input layer weights matrix using the text file.
      */
     public static void loadInputToFirstHiddenWeightMatrix() throws FileNotFoundException {
 
         // Prepare weights from between input layer -> hidden layer.
         File inputLayerWeightsFile = new File("InputToFirstHiddenWeights.txt");
         int rowPosition, columnPosition;
         Scanner fileScanner = new Scanner(inputLayerWeightsFile);
 
         for (rowPosition = 0; rowPosition < inputToFirstHiddenWeightMatrix.length; rowPosition++) {
             for (columnPosition = 0; columnPosition < inputToFirstHiddenWeightMatrix[rowPosition].length; columnPosition++) {
                 // Assign each float in the matrix to the float in the file.
                 inputToFirstHiddenWeightMatrix[rowPosition][columnPosition] = fileScanner.nextFloat();
             }
         }
 
         // Free up space.
         inputLayerWeightsFile = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads weights into the firsthiddentosecondhidden weights matrix using the text file.
      */
     public static void loadFirstHiddenToSecondHiddenWeightMatrix() throws FileNotFoundException {
 
         // Prepare weights from between input layer -> hidden layer.
         File firstHiddenToSecondHiddenWeightsFile = new File("FirstHiddenToSecondHiddenWeights.txt");
         int rowPosition, columnPosition;
         Scanner fileScanner = new Scanner(firstHiddenToSecondHiddenWeightsFile);
 
         for (rowPosition = 0; rowPosition < firstHiddenToSecondHiddenWeightMatrix.length; rowPosition++) {
             for (columnPosition = 0; columnPosition < firstHiddenToSecondHiddenWeightMatrix[rowPosition].length; columnPosition++) {
                 // Assign each float in the matrix to the float in the file.
                 firstHiddenToSecondHiddenWeightMatrix[rowPosition][columnPosition] = fileScanner.nextFloat();
             }
         }
 
         // Free up space.
         firstHiddenToSecondHiddenWeightsFile = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     /*
      * Parameters: None
      * 
      * Return value: None
      * 
      * Loads weights into the secondhiddentooutput weights matrix using the text file.
      */
     public static void loadSecondHiddenToOutputWeightMatrix() throws FileNotFoundException {
 
         // Prepare weights from between second hidden layer -> output layer.
         File secondHiddenToOutputWeightsFile = new File("SecondHiddenToOutputWeights.txt");
         int rowPosition, columnPosition;
         Scanner fileScanner = new Scanner(secondHiddenToOutputWeightsFile);
 
         for (rowPosition = 0; rowPosition < secondHiddenToOutputWeightMatrix.length; rowPosition++) {
             for (columnPosition = 0; columnPosition < secondHiddenToOutputWeightMatrix[rowPosition].length; columnPosition++) {
                 // Assign each float in the matrix to the float in the file.
                 secondHiddenToOutputWeightMatrix[rowPosition][columnPosition] = fileScanner.nextFloat();
             }
         }
 
         // Free up space.
         secondHiddenToOutputWeightsFile = null;
         fileScanner.close();
         fileScanner = null;
 
     }
 
     // Sets all activations in a layer to 0.0. Used for backpropagation to not mix up full runs between multiple files.
     public static void clearLayer(Neuron[] layer) {

        for (Neuron neuron : layer) {
            neuron.setActivation((float) 0.0);
        }

     }

     // Sets all activations in the network to 0.0.
     public static void clearNetwork() {

        clearLayer(inputLayer);
        clearLayer(firstHiddenLayer);
        clearLayer(secondHiddenLayer);
        clearLayer(outputLayer);

     }

 }