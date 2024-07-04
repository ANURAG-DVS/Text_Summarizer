🌟 Text Summarization using NLP 🌟


Group-3 Project Collaborators:  Anurag DVS, V. Srujana, Prabhjot Singh, Preeti Panjwani, Rupesh Sharda, Shreepad Raut
Abstractive Summarization with T5 Model
This project utilizes the T5 model, a state-of-the-art transformer-based architecture, to create summaries that may include rephrasing and new phrases not present in the original text.

Key Steps
1. Data Preparation 📦:
• Import necessary libraries. • Load the Samsum dataset, a benchmark for text summarization tasks.

2. Preprocessing 🔄:
• Tokenize the input articles and summaries to prepare them for the T5 model.

3. Model Loading 🚀:
• Load the pre-trained T5 model for conditional generation.

4. Summarization Function ✍️:
• Define a function that generates a concise summary using the T5 model.

5. Summarization Testing 🧪:
• Test the summarization function on sample articles and compare the generated summaries with the originals.

5. Evaluation 📈:
• Use the ROUGE metric to evaluate the quality of generated summaries.

6. Fine-Tuning 🔧:
• Provide an option to fine-tune the model using the Trainer class to customize specific tasks or datasets.

7. Re-evaluation 🔍:
• Re-evaluate the model using the ROUGE metric to gauge enhanced performance post-augmentation.

Results
📊 Comprehensive evaluation results demonstrate the T5 model's proficiency in summarizing text across various metrics. Explore the notebook to understand the intricacies of text summarization and harness the power of the T5 model!

Extractive Summarization with spaCy
This project uses spaCy, a robust NLP library, to extract meaningful summaries by identifying and selecting the most important sentences from the original text.

Key Steps
1. Dependency Installation 🔧:
• Install necessary libraries, including spaCy and tabulate. • Download the English language model for spaCy.

2. Text Processing 🔄:
• Define and process the text to be summarized. • Tokenize the text and calculate word frequencies.

3. Sentence Scoring 📊:
• Score each sentence based on the frequencies of the words it contains.

4. Summary Generation ✍️:
• Generate the extractive summary by selecting the top sentences.

5. Result Formatting 🗂️:
Format the summary and original text into a structured table using the tabulate library.

Results
📊 The project generates extractive summaries that capture the essence of the original text. The results are displayed in a structured table format for clarity. The summary model is saved as a pkl file for easy integration into web applications.
