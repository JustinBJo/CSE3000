# Tutorial 2 - Bayes’ Rule

## After this tutorial, you will be able to:

- Apply Bayes' rule to solve probability problems
- Understand the relationship between prior, likelihood, and posterior probabilities
- Calculate conditional probabilities in real-world scenarios
- Evaluate decisions with Bayesian reasoning

## Chapter 1: Introduction to Bayes' Rule

### 1.1 What is Bayes' Rule?

Bayes' rule (also called Bayes' theorem) is a fundamental principle in probability theory that describes how to update our beliefs about events when we receive new evidence. It provides a mathematical framework for combining prior knowledge with new data.

The formula for Bayes' rule is:

$$
P(A|B) = \frac{P(B|A) × P(A)}{P(B)}
$$

Where:

- P(A|B) is the posterior probability (probability of A given B)
- P(B|A) is the likelihood (probability of B given A)
- P(A) is the prior probability (initial probability of A)
- P(B) is the marginal probability (total probability of B)

### 1.2 Real-World Example: Medical Diagnosis

Let's consider a doctor diagnosing a rare disease:

1. Prior knowledge: The disease affects 1% of the population [P(D) = 0.01]
2. Test accuracy:
    - 95% positive if you have the disease [P(+|D) = 0.95]
    - 90% negative if you don't have it [P(-|H) = 0.90]

If a patient tests positive, what's the probability they have the disease?

Using Bayes' rule:

$$
P(D|+) = \frac{P(+|D) × P(D)}{P(+)}
$$

$$
= \frac{(0.95 * 0.01)}{(0.95 * 0.01) + (0.10  * 0.99)}
$$

$$
≈ 0.087, 8.7\%
$$

## Chapter 2: Components of Bayes' Rule

### 2.1 Prior Probability

Prior probability represents our initial belief about an event before seeing new evidence. It's based on:

- Historical data
- Previous experience
- General knowledge
- Initial assumptions

### 2.2 Likelihood

Likelihood represents how probable the evidence is, given our hypothesis:

- Measures the compatibility of the evidence with different hypotheses
- Often based on empirical data or scientific models
- Can be updated as more data becomes available

### 2.3 Posterior Probability

Posterior probability is our updated belief after considering the evidence:

- Combines prior knowledge with new evidence
- Becomes the new prior for future updates
- Represents our current best estimate

## Chapter 3: Bayesian Classification and Error

### 3.1 Bayesian Classification

Bayesian classification uses Bayes' rule to make decisions by comparing posterior probabilities:

- Choose class ω₁ if P(ω₁|x) > P(ω₂|x)
- Choose class ω₂ otherwise

Where:

- ω₁, ω₂ are possible classes
- x is the observed data/features

### 3.2 Classification Error

Two types of errors can occur:

1. **Type I Error (False Positive)**
    - Incorrectly classifying as ω₁ when true class is ω₂
    - Error probability: P(decide ω₁|ω₂)
2. **Type II Error (False Negative)**
    - Incorrectly classifying as ω₂ when true class is ω₁
    - Error probability: P(decide ω₂|ω₁)
    
    ![image.png](Tutorial%202%20-%20Bayes%E2%80%99%20Rule%2013e92a4820738053a5a1d512f353868b/image.png)
    

### 3.3 Bayes Error Rate

The Bayes error rate is the theoretical minimum error achievable by any classifier:

- For two classes: Error = min[P(ω₁|x), P(ω₂|x)]
- Represents inherent overlap between classes
- Cannot be eliminated even with perfect classification

Example:
Given two overlapping distributions:

- P(x|ω₁) = N(1, 1) // Normal distribution, mean=1, variance=1
- P(x|ω₂) = N(2, 1) // Normal distribution, mean=2, variance=1
- P(ω₁) = P(ω₂) = 0.5

![image.png](Tutorial%202%20-%20Bayes%E2%80%99%20Rule%2013e92a4820738053a5a1d512f353868b/image%201.png)

The Bayes error rate would be the area where the wrong class has higher probability.

### 3.4 Cost of Errors

Different errors may have different costs:

- Medical diagnosis: False negatives more costly than false positives
- Spam detection: False positives more problematic than false negatives

Decision rule with costs:

- Choose ω₁ if P(ω₁|x)C(ω₂|ω₁) > P(ω₂|x)C(ω₁|ω₂)
- Where C(ωᵢ|ωⱼ) is cost of deciding class i when true class is j

## Conclusion

You've learned:

1. The fundamental components of Bayes' rule
2. How to apply it to real-world problems
3. Best practices for Bayesian reasoning
