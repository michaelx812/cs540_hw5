import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.HashSet;
import java.util.HashMap;

/**
 * Your implementation of a naive bayes classifier. Please implement all four methods.
 */

public class NaiveBayesClassifier implements Classifier {
    private Map<Label,Integer> docPerLabel;
    private Map<Label,Integer> wordPerLabel;
    
    private Map<String,Integer> wordsOfPos;
    private Map<String,Integer> wordsOfNeg;

    private Set<String> dict;
    
    private double positivePrior;
    private double negativePrior;

    private int vocab;
    /**
     * Trains the classifier with the provided training data and vocabulary size
     */
    @Override
    public void train(List<Instance> trainData, int v) {
        //count of vocabulary
        vocab = v;
        // Hint: First, calculate the documents and words counts per label and store them. 
        docPerLabel = getDocumentsCountPerLabel(trainData);
        wordPerLabel = getWordsCountPerLabel(trainData);
        // Then, for all the words in the documents of each label, count the number of occurrences of each word.
        dict = new HashSet<String>();
        for(Instance inst: trainData){
            if(inst.label == Label.POSITIVE){
                for(String s: inst.words()){
                    dict.add(s);
                    wordsOfPos.put(s,wordsOfPos.getOrDefault(s,0)+1);
                }
            }else if(inst.label == Label.NEGATIVE){
                for(String s: inst.words()){
                    dict.add(s);
                    wordsOfNeg.put(s,wordsOfNeg.getOrDefault(s,0)+1);
                }
            }
        }
        // Save these information as you will need them to calculate the log probabilities later.
        int N = trainData.size();
        if(N==0){
            negativePrior = 0.0;
            positivePrior = 0.0;
        }else{
            positivePrior = (double)docPerLabel.getOrDefault(Label.POSITIVE, 0)/(double)N;
            negativePrior = (double)docPerLabel.getOrDefault(Label.NEGATIVE, 0)/(double)N;
        }
        //
        // e.g.
        // Assume m_map is the map that stores the occurrences per word for positive documents
        // m_map.get("catch") should return the number of "catch" es, in the documents labeled positive
        // m_map.get("asdasd") would return null, when the word has not appeared before.
        // Use m_map.put(word,1) to put the first count in.
        // Use m_map.replace(word, count+1) to update the value
    }

    /*
     * Counts the number of words for each label
     */
    @Override
    public Map<Label, Integer> getWordsCountPerLabel(List<Instance> trainData) {
        Map<Label,Integer> map = new HashMap<>();
        int pCounts = 0, nCounts = 0;
        for(Instance inst: trainData){
            if(inst.label==Label.POSITIVE){
                for(String s: inst.words){
                    pCounts++;
                }
            }else if(inst.label==Label.NEGATIVE){
                for(String s: inst.words){
                    nCounts++;
                }
            }
        }
        map.put(Label.POSITIVE,pCounts);
        map.put(Label.NEGATIVE,nCounts);
        return map;
    }


    /*
     * Counts the total number of documents for each label
     */
    @Override
    public Map<Label, Integer> getDocumentsCountPerLabel(List<Instance> trainData) {
        Map<Label,Integer> map = new HashMap<>();
        for(Instance inst: trainData){
            if(inst.label==Label.POSITIVE){
                map.put(Label.POSITIVE,map.getOrDefault(1, 0)+1);
            }else if(inst.label==Label.NEGATIVE){
                map.put(Label.NEGATIVE,map.getOrDefault(1, 0)+1);
            }
        }
        return map;
    }


    /**
     * Returns the prior probability of the label parameter, i.e. P(POSITIVE) or P(NEGATIVE)
     */
    private double p_l(Label label) {
        if(label == Label.POSITIVE){
            return positivePrior;
        }else if(label == Label.NEGATIVE){
            return negativePrior;
        }
        return 0;
    }

    /**
     * Returns the smoothed conditional probability of the word given the label, i.e. P(word|POSITIVE) or
     * P(word|NEGATIVE)
     */
    private double p_w_given_l(String word, Label label) {
        // Calculate the probability with Laplace smoothing for word in class(label)
        double numerator=0.0;
        double denominator=0.0;

        Map<String,Integer> map;
        if(label == Label.POSITIVE){
            map = wordsOfPos;
        }else{
            map = wordsOfNeg;
        }
        numerator = (double)map.getOrDefault(word, 0) + 1.0;


        for(String s: dict){
            denominator += (double)map.getOrDefault(s, 0);
        }
        denominator += (double)vocab;

        return denominator == 0.0 ? 0.0:numerator/denominator;
    }

    /**
     * Classifies an array of words as either POSITIVE or NEGATIVE.
     */
    @Override
    public ClassifyResult classify(List<String> words) {
        // TODO : Implement
        // Sum up the log probabilities for each word in the input data, and the probability of the label
        // Set the label to the class with larger log probability
        double posLog = positivePrior == 0.0 ? (double)Integer.MIN_VALUE:Math.log(positivePrior);
        double negLog = negativePrior == 0.0 ? (double)Integer.MIN_VALUE:Math.log(negativePrior);

        for(String s: words){
            double posCond = p_w_given_l(s, Lable.POSITIVE);
            double negCond = p_w_given_l(s, Label.NEGATIVE);

            posLog += posCond == 0 ? (Double)Integer.MIN_VALUE:Math.log(posCond);
            negLog += negCond == 0 ? (Double)Integer.MIN_VALUE:Math.log(negCond);
        }
        ClassifyResult result = new ClassifyResult();
        if(posLog<negLog){
            result.label = Label.NEGATIVE;
        }else{
            result.label = Label.POSITIVE;
        }

        Map<Label, Double> logs = new Map<Label, Double>();
        logs.put(Label.POSITIVE,posLog);
        logs.put(Label.NEGATIVE,negLog);

        result.logProbPerLabel = logs;
        return result;
    }


}
