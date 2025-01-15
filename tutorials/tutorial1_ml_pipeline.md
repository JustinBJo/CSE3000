# Tutorial 1 - Machine Learning Pipeline

## After this tutorial, you will be able to:

- Understand the general machine learning pipeline
- Explain the purpose of training, test and validation sets
- Compare the performance of a model trained on different dataset splits
- Identify overfitting and underfitting of a machine learning model

## Chapter 1: Introduction to the Machine Learning Pipeline

### **1.1 What is a Machine Learning Pipeline?**

In order to create and use ML models such as facial recognition, voice-to-text, or Instagram filters, we need to train and test the model. A machine learning pipeline is the series of steps involved in building, training, and evaluating a machine learning model. It organizes the process to ensure the model learns effectively from the data and generalizes well to unseen situations. In short, we prepare the resource and give it to the model to learn, and we test how well can it apply it to other examples.

Below are the stages in the ML Pipeline:

![image.png](Tutorial%201%20-%20Machine%20Learning%20Pipeline%2013e92a48207380729045c1b34e684668/image.png)

1. **Data Collection**: Gathering data relevant to the problem you want to solve (e.g., images or videos for training an Instagram filter).
2. **Data Preparation**: Cleaning and organizing the data for analysis (e.g., cropping images, resizing them, or converting them to grayscale).
3. **Model Training**: Using a part of the data to teach the model how to recognize or enhance patterns, such as facial features.
4. **Model Evaluation**: Testing the model's performance on unseen data to ensure it works well on photos or videos not included in training.

### **1.2 Real-World Example: Instagram Filters**

Let’s consider Instagram filters, which use machine learning to enhance photos or apply effects in real-time.

1. **Data Collection**:
    - Collect thousands of labeled images, such as photos of faces, objects, or landscapes.
2. **Data Preparation**:
    - Preprocess images: Crop them to focus on the subject, normalize the colors, or adjust brightness.
    - Label data: Assign categories like “faces,” “scenery,” or “objects” to help the model learn.
3. **Model Training**:
    - Train the model to detect features like eyes, mouth, and face shape using the labeled images.
    - The model learns to apply effects (e.g., smoothing skin, adding virtual sunglasses) to these features.
4. **Model Evaluation**:
    - Test the model on new images to ensure it recognizes faces correctly and applies the filter realistically, even for unseen photos.

This pipeline enables Instagram filters to work seamlessly on millions of users' photos, adapting to different faces, lighting conditions, and angles.

## Chapter 2: Overfitting and Underfitting

### **2.1 What Is Overfitting?**

Overfitting occurs when a model learns the training data too well, including noise or specific details that do not generalize to new data.

If a filter works perfectly on the faces in the training set but fails to apply effects correctly on new faces, this filter is overfitting. This can be due to over-reliance on training data specifics.

### **2.2 What Is Underfitting?**

Underfitting occurs when a model fails to learn enough from the training data, leading to poor performance on both training and test sets.

If a filter does not detect facial features like eyes or mouths correctly, this is underfitting. This can be caused because the model has not captured the essential patterns during training.

![image.png](Tutorial%201%20-%20Machine%20Learning%20Pipeline%2013e92a48207380729045c1b34e684668/image%201.png)

*source: [https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76)*

## Chapter 3: Training and Test Sets

### **3.1 What are Training and Test Sets?**

In the ML pipeline, data is split into two main parts:

1. **Training Set**: This is the data the model uses to learn patterns. For example, Instagram filters are trained on thousands of labeled photos, teaching the model to recognize features like eyes, noses, or mouths.
2. **Test Set**: This is a separate set of data used to evaluate how well the model performs on new, unseen photos.

### **3.2 Why Do We Need Training and Test Sets?**

Using separate training and test sets ensures that the model can generalize to new data. If a model is only evaluated on the data it was trained on, we risk overestimating its performance.

This is just like when you study for an exam. If you study using a practice exam and only test yourself with questions on this exam, you are likely to get a good result because you have seen the questions before. However, it is not guaranteed that you will do good on the final test which has the questions you have never seen before. This is why we need a test set; we need to make sure we perform well for any new questions.

Then given a dataset, what would be a good division for the training and test sets?

### **3.3 How Do We Decide the Sizes of Training and Test Sets?**

When deciding how to split data into two sets, we need to consider various factors such as the size of the database, type of the model, etc.

![image.png](Tutorial%201%20-%20Machine%20Learning%20Pipeline%2013e92a48207380729045c1b34e684668/image%202.png)

![image.png](Tutorial%201%20-%20Machine%20Learning%20Pipeline%2013e92a48207380729045c1b34e684668/image%203.png)

Big training set will ensure that the model is trained with less bias, but if the test set is too small it will be hard to test if the model is reliable.

This introduces the risk of **overfitting.**

Big test set will make sure that the model is tested well, but if the training set is too small it will result in poorly trained model, making it impossible to use.

This introduces the risk of **underfitting.**

Some widely used data division are:

- **80-20 split**: Common for small to medium datasets.
- **90-10 split**: Useful for large datasets where even a small test set is representative.
- **70-30 split**: Occasionally used when robust evaluation is critical.

- Large datasets:
    - Example: 10,000 images. Use 9,000 for training and 1,000 for testing (90-10 split).
    - This ensures enough data for evaluation without sacrificing training quality.
- Small datasets:
    - Example: 1,000 images. Use 800 for training and 200 for testing (80-20 split).
    - A smaller training set requires more careful balance to avoid underfitting.

## Chapter 4: Validation Set

### **4.1 What is a Validation Set?**

We have seen that when building a machine learning model, the ultimate goal is to create a system that works well not just on the training data but also on new, unseen data. To achieve this with even more efficiency, we introduce a new part of the dataset called **validation set.** 

A validation set is like a "progress check" during training. It helps prevent overfitting by showing us when the model might be learning the training data too perfectly.

### **4.2 Real-World Example: Instagram Filters**

When developing Instagram filters, the validation set might contain a small subset of unseen photos during training.

1. The model learns from the training set to recognize facial features.
2. Periodically, its performance is tested on the validation set to check if it applies filters correctly to new faces.
3. Developers might adjust hyperparameters, such as the size of the filter or the way facial landmarks are detected, based on validation set results.

### **4.3 Splitting Data into Training, Validation, and Test Sets**

Incorporating a validation set requires splitting the data into three parts:

![image.png](Tutorial%201%20-%20Machine%20Learning%20Pipeline%2013e92a48207380729045c1b34e684668/image%204.png)

1. **Training Set**: 60–70% of the data.
2. **Validation Set**: 10–20% of the data.
3. **Test Set**: 10–20% of the data.

If you have 1,000 images, you can use:

- 600 for training.
- 200 for validation during development.
- 200 for final testing.

## Conclusion

Congratulations! You have completed this tutorial. Here’s a recap of what you’ve learned:

1. **Understanding the ML Pipeline**:
    
    You now know how data flows through the stages of collection, preparation, training, and evaluation to create robust machine learning models.
    
2. **Overfitting and Underfitting**:
    
    You’ve learned about the balance between a model that’s too specific (overfitting) and one that’s too simplistic (underfitting) and how data management helps mitigate these issues.
    
3. **Training, Test, and Validation Sets**:
    
    You’ve explored how datasets are split into training, test, and validation sets, and how each serves a distinct and vital role in building and testing models.
    
4. **Deciding Train-Test-Validation Sizes**:
    
    You’ve understood how the size of these dataset splits can impact a model’s ability to learn and generalize effectively.
