# Readability Chat Bot

RoBERTa-Base transformer model is utilised and trained on a regression task with a dataset of various English text excerpts to predict a continuous score correlating to the apparent difficulty of the text (readability). The model was then deployed as an API.

The NLP model was tested on 100 popular texts extracted from the Project Gutenberg website.

A language understanding chat bot, utilising entity and intent recognition was also developed as a means of assisting in determining the appropriate reading level of texts. The chat bot was interfaced with an Azure ML API, to return scores based on user-inputted texts.

* Files for the model deployment and training can be found in the 'deployment-files' folder
* Files for web scraping on Project Gutenberg can be found in the 'gutenberg-scrape' folder
* Source files for the chatbot can be found in the 'commonlit_bot' folder

### Dataset

This project was inspired by a [Kaggle competition](https://www.kaggle.com/c/commonlitreadabilityprize), and a dataset was pre-formatted on Kaggle with the format of: 

| **excerpt** | **target**  |
| ----------- | ----------- |
| "When the young people returned to the ballroom, it presented a decidedly changed appearance. Instead of an interior scene, it was a winter landscape. The floor was covered with snow-white canvas, not laid on smoothly, but rumpled over bumps and hillocks, like a real snow field. The numerous palms and evergreens that had decorated the room, were powdered with flour and strewn with tufts of cotton, like snow. Also diamond dust had been lightly sprinkled on them, and glittering crystal icicles hung from the branches. At each end of the room, on the wall, hung a beautiful bear-skin rug. These rugs were for prizes, one for the girls and one for the boys. And this was the game. The girls were gathered at one end of the room and the boys at the other, and one end was called the North Pole, and the other the South Pole. Each player was given a small flag which they were to plant on reaching the Pole. This would have been an easy matter, but each traveller was obliged to wear snowshoes." | -0.340259125 | 
| All through dinner time, Mrs. Fayre was somewhat silent, her eyes resting on Dolly with a wistful, uncertain expression. She wanted to give the child the pleasure she craved, but she had hard work to bring herself to the point of overcoming her own objections. At last, however, when the meal was nearly over, she smiled at her little daughter, and said, "All right, Dolly, you may go." "Oh, mother!" Dolly cried, overwhelmed with sudden delight. "Really? Oh, I am so glad! Are you sure you're willing?" "I've persuaded myself to be willing, against my will," returned Mrs. Fayre, whimsically. "I confess I just hate to have you go, but I can't bear to deprive you of the pleasure trip. And, as you say, it would also keep Dotty at home, and so, altogether, I think I shall have to give in." "Oh, you angel mother! You blessed lady! How good you are!" And Dolly flew around the table and gave her mother a hug that nearly suffocated her. | -0.315372342 |

### Model

* Tokenization of text was performed using pre-trained [RoBERTa-Base tokenizer](https://huggingface.co/transformers/model_doc/roberta.html#robertatokenizerfast)
* A [RoBERTa-Base](https://huggingface.co/transformers/model_doc/roberta.html#robertaforsequenceclassification) regression model was trained via the Huggingface API to predict the target scores from the text.

### Model evaluation

* The model was evaluated on 100 randomly generated excerpts from the Top 100 texts on Project Gutenberg

![image](https://user-images.githubusercontent.com/65014987/138953355-a351568b-a50b-434f-87cb-b40e2d3a5945.png)
*Figure 1: Top 14 texts from Project Gutenberg as of October 2021*

### Readability Bot

Language understanding chat bot was developed via the Microsoft Bot Framework Composer, utilising Azure resources to support language understanding. To configure language understanding, one must create their own LUIS authoring endpoint via Azure, and input the authoring key into Bot Composer as shown below. 

![image](https://user-images.githubusercontent.com/65014987/138960106-fb7a63b7-a5f0-4879-8058-95a64b7eea45.png)

Hosting and publishing the bot requires a profile to be generated with the required cloud resources 

![image](https://user-images.githubusercontent.com/65014987/138960418-d40b40fc-18c3-4cd3-a252-4adbcc67b9de.png)

**For info on how to use the chat bot and its functionalities, visit this [link](https://github.com/rl16432/Readability-Bot/tree/master/commonlit_bot).**

Link to the chatbot hosted on Telegram: [https://t.me/Commonlit_Bot](https://t.me/Commonlit_Bot)

*Note:* The API and chat bot's functionalities are disabled due to cloud resource costs