# Disaster Response Pipeline Project

The Disaster Response Pipeline Project aims to build a web application that can classify disaster-related messages into different categories, enabling quick and effective response during emergencies. The project involves data processing, machine learning model training, and a user-friendly web interface to interact with the model.

### Instructions:

Follow these steps to set up your database, train the model, and run the web app.

1. To run the ETL pipeline that cleans and stores data in the database:
   ```shell 
   python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
   ```
   
2. To run the ML pipeline that trains the classifier and saves it:
   ```shell
   python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
   ```

3. Go to `app` directory: 
   ```shell 
   cd app
   ```

4. Run your web app: 
   ```shell 
   python run.py
   ```

5. Once the app is running, open your web browser and go to: 
   ```browser 
   http://localhost:3001
   ```

### Project Structure
- **app:** Contains the Flask web app files.  
    - **templates:** holds HTML templates for web pages.  
    - **run.py** is the script to run the web app.
   
   
- **data:** Contains the data processing script and input data files.  
    - **process_data.py** cleans and stores data in the database.  
    - **disaster_messages.csv** contains message data.  
    - **disaster_categories.csv** contains category data.  


- **models:** Contains the machine learning script and the trained model.
    - **train_classifier.py** trains and saves the classification model.
    - **classifier.pkl** is the trained model saved as a pickle file.

### Usage
The homepage displays visualizations related to disaster message data.  
Users can enter a message in the input form to classify its category based on the trained model.

### Acknowledgments
This project is part of the Udacity Data Science Nanodegree program.
