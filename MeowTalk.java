// Java JDK
import java.util.Scanner;
import javax.sound.sampled.*;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;

/*
 * NOTES:
 * 1. The sample rate, 2.4 KHz, is chosen to accommodate for the Shannon-Nyquist frequency of 1.2 KHz, the maximum vocal range of
 *      a cat (Needs to be cited).
 * 2. Keyword "parameter" means the collection of weight and biases the program uses to alter signals that are propagated
 *      forward throughout the neural network.
 * 3. Sigmoid function is currently not in use and the program will output large values. I will implement it again once
 *      I find an activation function that proportionally scales the data from 0.0 to 1.0.
 */

public class MeowTalk {

    public static void main(String[] args) throws IOException {

        //Variables
        Scanner input = new Scanner(System.in);
        File file = null;
        AudioInputStream audioInputStream = null;

        // Load parameters using text files.
        System.out.println("The program is loading in all of its necessary files; your patience is appreciated."
                            + "\nLoading...");

        try {
            NeuralNetwork.loadParameters();
        }
        catch (FileNotFoundException exception) {
            System.out.println("Sorry, it seems this program is missing one or more of its necessary files.");
            System.exit(0);
        }

        // Introduction and warning of program use.
        Help.about();

        // If the user enters "train", the program goes into a sort of debug mode that begins training the network.
        // Otherwise, get to the regular purpose of the program.
        if (input.nextLine().equalsIgnoreCase("train")) {

            // Notify user
            System.out.println("\nTHE PROGRAM HAS ENTERED TRAINING MODE."
                                + "\nIF THE USER DID NOT INTEND THIS, PLEASE ENTER \"exit\" INTO THE COMMAND LINE."
                                + "\nIF THIS WAS INTENDED, ENTER ANYTHING ELSE TO CONTINUE THE TRAINING MODE.\n");

            // Expect input from user, if the user does not enter "exit", the program continues toward training the network.
            if (!input.nextLine().equalsIgnoreCase("exit")) {

                boolean fileExists = false;
                String batchName = null;
                do {

                    System.out.println("ENTER A NAME OF THE BATCH FILE TO TRAIN THE NETWORK WITH.\n");

                    batchName = input.nextLine();

                    File batch = new File("trainingdatafiles\\" + batchName);

                    if (!batch.exists()) {
                        System.out.println("\nTHAT IS NOT A VALID NAME.");
                    }
                    else {
                        fileExists = true;
                    }

                } while (!fileExists);
                
                // Method used to encapsulate the work needed to train the network, using a batch of training data.
                NeuralNetwork.trainNetwork(batchName);

                // Close input because program will close after training anyway.
                input.close();
                return;

            }
            else {

                System.out.println("\nEXITING TRAINING MODE.");

            }

        }

        System.out.println("Thank you!");
        System.out.println("Please wait while we finish setting up the recording software...");

        //TargetDataLine specifications
        TargetDataLine targetDataLine = null;
        AudioFormat.Encoding encoding = AudioFormat.Encoding.PCM_SIGNED; // Audio encoding technique
        float sampleRate = 2400; // Number of samples per second.
        int sampleSizeInBits = 16; // Number of bits in each sample
        int channels = 1; // Number of channels (mono in this instance)
        int frameSize = 2; // Number of bytes in each frame (channels * sampleSizeInBits / 8)
        float frameRate = 2400; // Number of frames per second (sampleRate essentially)
        boolean bigEndian = true; // I don't even want to pretend like I know what this means

        // Create new audioformat object to get the correct data line.
        try {
            AudioFormat audioFormat = new AudioFormat(encoding, sampleRate, sampleSizeInBits, channels, frameSize, frameRate, bigEndian);
            DataLine.Info info = new DataLine.Info(TargetDataLine.class, audioFormat);

            if (!AudioSystem.isLineSupported(info)) {
                System.out.println("Sorry, your computer's audio system is not capable of running this software.");
                System.exit(0);
            }

            //Create target line for audio input.
            targetDataLine = (TargetDataLine) AudioSystem.getLine(info);

            //Notify user that program is about to record audio.
            //Expecting input from user...
            System.out.println("...Success.");
            System.out.println("Get your cat close to your microphone, and enter any key/phrase to begin recording.");
            System.out.println("Once an input is made, the program will record for exactly five seconds.");
            input.nextLine();
            System.out.println(""); //Create a new line to organize output.

            //Prepare to record audio and write data into a new file to be processed later.
            targetDataLine.open();
            audioInputStream = new AudioInputStream(targetDataLine);
            file = new File("recordedAudio.wav");
            file.createNewFile();

            //Begin writing incoming audio to file.
            targetDataLine.start();

            //Creates a thread to allow the program to wait to stop the stream after five seconds. That's pretty cool.
            AudioRecordThread<AudioInputStream> audioRecordThread = new AudioRecordThread<>(audioInputStream, AudioFileFormat.Type.WAVE, file);
            audioRecordThread.start();
            
            // Pause main thread for five seconds (5000 milliseconds), and then stops streaming.
            try {
                Thread.sleep(5000);
            }
            catch (InterruptedException exception) {
                System.out.println(exception);
            }

            //Clean things up a bit.
            targetDataLine.stop();
            targetDataLine.close();
            
        }
        catch (LineUnavailableException exception) {
            System.out.println("Sorry, your computer's audio system is not capable of running this software.");
            System.out.println("This program will now close. Please run the program again if you think that will fix it.");
            System.exit(0);
        }

        // Begin using file to prepare neural network.
        try {
            NeuralNetwork.prepareInputLayer(file);
        }
        catch (Exception exception) {
            System.out.println(exception);
        }

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

        //Print calculated expression
        NeuralNetwork.printCalculatedExpression();

        // Delete file to reduce clutter (Currently doesn't work for an unknown reason; the file is virtually immortal);
        file.delete();

        // Eventually close input... eventually...
        input.close();

        System.exit(0);

    }

}

//Class that is an extension of thread. Used to allow the main thread to wait for five seconds while the system is recording audio.
class AudioRecordThread<E extends AudioInputStream> extends Thread {

    private E stream;
    private AudioFileFormat.Type type;
    private File file;

    //Constructor: Pass-by-reference values don't matter because changing the object won't occur... maybe?
    public AudioRecordThread(E stream, AudioFileFormat.Type type, File file) {
        super(); //Call superclass (that's Thread!) default constructor.
        this.stream = stream;
        this.type = type;
        this.file = file;
    }

    //Record cat speech, wait for user input to end recording.
    @Override public void run() {

        System.out.println("Your system is now recording audio. Make some noise!");
        
        try {
            AudioSystem.write(stream, type, file); //File type should not be a problem currently.
        }
        catch (IOException exception) {
            System.out.println("Sorry, this program detected a problem with audio files: " + exception.getMessage());
        }

        System.out.println("Cat speech successfully stopped recording."); //Notify user that thread has ended.

    }

}