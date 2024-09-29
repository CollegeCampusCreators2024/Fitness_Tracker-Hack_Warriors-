Team = Hack Warriors
Members:
1. Sahil Banerjee
2. Shivay Bhargava
3. Shubham Barawal
4. Krishna Soni






Setting Up the Environment
Twilio API Setup:

Sign up for a Twilio account if you haven't already.
Create a new project to get your Account SID, Auth Token, and a Twilio phone number.
Nutritionix API Setup:

Sign up for a Nutritionix account and create an application to get your App ID and API Key.
Google Sheets Setup:

Create a new Google Sheet and share it with the email linked to your Google account.
Use the Sheety API to obtain the necessary URL for logging data.
Update API Keys:

Replace the placeholder values in the script with your actual Twilio, Nutritionix, and Sheety API credentials.

Running the Application
Save the script in a Python file (e.g., exercise_counter.py).
Open your terminal and navigate to the directory where the script is located.
Run the application using Streamlit:
A new tab will open in your default web browser displaying the application interface.

Using the Application
User Details: In the sidebar, input your exercise details:

Select the exercise (Bicep Curls or Lateral Raises).
Adjust the sliders to input your weight, height, age, and select your gender.
Start Exercise: Click the "Start Exercise" button to begin counting repetitions. Ensure your camera is correctly positioned to capture your movements.

Stop Exercise: Click the "Stop Exercise" button to stop counting and send the final SMS notification.

Logging: Every 15 repetitions, the application will log your exercise data to Google Sheets and send an SMS notification.
